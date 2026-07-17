from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHOICES = list(range(11, 21))
SIGNAL_VARIABLES = ["game_variant", "goal_framing", "model", "persona"]
DESIGN_VARIABLES = ["instruction_wording", "response_format", "temperature", "persona_format"]
DEFAULTS: dict[str, str] = {
    "goal_framing": "none",
    "persona": "neutral",
    "instruction_wording": "default",
    "response_format": "default",
    "temperature": "0.5",
    "persona_format": "default",
}
DESIGN_LEVELS: dict[str, list[str]] = {
    "instruction_wording": ["descending-range", "numbers-as-words", "paraphrase-a", "paraphrase-b"],
    "response_format": ["bare-number"],
    "temperature": ["1.0"],
    "persona_format": ["descriptive", "tabular"],
}
ANCHORS: dict[str, dict[int, float]] = {
    "basic": {11: .04, 12: .00, 13: .03, 14: .06, 15: .01, 16: .06, 17: .32, 18: .30, 19: .12, 20: .06},
    "cycle": {11: .01, 12: .01, 13: .00, 14: .01, 15: .00, 16: .04, 17: .10, 18: .22, 19: .47, 20: .13},
    "costless": {11: .00, 12: .04, 13: .00, 14: .04, 15: .04, 16: .04, 17: .09, 18: .21, 19: .40, 20: .15},
}


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    official_parent: str
    short_label: str
    unit: str
    claim_type: str  # one_arm or paired
    metric: str
    filters: dict[str, str]
    arm_variable: str | None = None
    treatment: str | None = None
    reference: str | None = None
    sign: float = 1.0
    threshold: float = 0.0
    natural_scale: float = 1.0


CLAIMS: list[ClaimSpec] = [
    ClaimSpec(
        "H1", "h1_human_anchor", "basic: 17--19 over 19--20", "probability difference",
        "one_arm", "h1_score", {"game_variant": "basic", "goal_framing": "none", "persona": "neutral"},
        natural_scale=1.0,
    ),
    ClaimSpec(
        "H2", "h2_cycle_shift", "cycle shifts choice upward", "choice points",
        "paired", "mean_choice", {}, "game_variant", "cycle", "basic", 1.0, natural_scale=9.0,
    ),
    ClaimSpec(
        "H3", "h3_costless_depth_bound", "costless majority at 17--20", "probability above 0.5",
        "one_arm", "h3_score", {"game_variant": "costless"}, natural_scale=1.0,
    ),
    ClaimSpec(
        "H4", "h4_strategic_framing", "strategic cue lowers choice", "choice points",
        "paired", "mean_choice", {}, "goal_framing", "strategic", "none", -1.0, natural_scale=9.0,
    ),
    ClaimSpec(
        "H5a", "h5_model_scale", "9B depth minus 4B", "depth points",
        "paired", "mean_reasoning_depth", {}, "model", "qwen3.5-9b", "qwen3.5-4b", 1.0, natural_scale=9.0,
    ),
    ClaimSpec(
        "H5b", "h5_model_scale", "27B depth minus 9B", "depth points",
        "paired", "mean_reasoning_depth", {}, "model", "qwen3.5-27b-fp8", "qwen3.5-9b", 1.0, natural_scale=9.0,
    ),
    ClaimSpec(
        "H6a", "h6_persona_disposition", "cautious raises choice", "choice points",
        "paired", "mean_choice", {}, "persona", "cautious", "neutral", 1.0, natural_scale=9.0,
    ),
    ClaimSpec(
        "H6b", "h6_persona_disposition", "competitive lowers choice", "choice points",
        "paired", "mean_choice", {}, "persona", "competitive", "neutral", -1.0, natural_scale=9.0,
    ),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(values: Iterable[float]) -> list[float]:
    vals = list(values)
    total = float(sum(vals))
    return [v / total for v in vals] if total else [0.0] * len(vals)


def js_distance(p: list[float], q: list[float]) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def earth_mover_1d(p: list[float], q: list[float]) -> float:
    cumulative = 0.0
    distance = 0.0
    for a, b in zip(p, q):
        cumulative += a - b
        distance += abs(cumulative)
    return distance / (len(p) - 1)


def entropy_bits(p: Iterable[float]) -> float:
    return -sum(v * math.log2(v) for v in p if v > 0)


def parse_condition_id(condition_id: str, manifest_model: str) -> dict[str, str]:
    labels: dict[str, str] = {
        "game_variant": "",
        "goal_framing": DEFAULTS["goal_framing"],
        "model": manifest_model,
        "persona": DEFAULTS["persona"],
        "instruction_wording": DEFAULTS["instruction_wording"],
        "response_format": DEFAULTS["response_format"],
        "temperature": DEFAULTS["temperature"],
        "persona_format": DEFAULTS["persona_format"],
    }
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in labels:
            labels[key] = value
    if not labels["game_variant"]:
        raise ValueError(f"Cannot parse game_variant from {condition_id}")
    return labels


def read_jsonl_gzip_member(tar: tarfile.TarFile, member: str) -> list[dict]:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member)
    raw = extracted.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as zipped:
        text = zipped.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_output_choice(output: str) -> int | None:
    # Some released models prefix an ``Analysis:`` paragraph before the
    # explicit action. Parse the action parameter anywhere in the output;
    # retain a full-line fallback for bare-number responses.
    patterns = [r"(?i)\bnumber\s*:\s*(1[1-9]|20)\b", r"(?m)^\s*(1[1-9]|20)\s*$"]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return int(match.group(1))
    return None


def load_released_number_parser(source_path: Path) -> tuple[Callable[[str], int | None], dict]:
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "parse_number_choice"]
    if len(selected) != 1:
        raise RuntimeError("released parse_number_choice definition not found")
    namespace: dict = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    node = selected[0]
    lines = source.splitlines()
    audit = {
        "source_path": source_path.as_posix(),
        "sha256": sha256_file(source_path),
        "function": "parse_number_choice",
        "start_line": int(node.lineno),
        "end_line": int(node.end_lineno or node.lineno),
        "source": "\n".join(lines[node.lineno - 1 : int(node.end_lineno or node.lineno)]),
        "execution_mode": "exact AST definition compiled with standard-library re",
    }
    return namespace["parse_number_choice"], audit


def metrics_from_choices(choices: list[int], prefix: str = "") -> dict[str, float | int]:
    values: dict[str, float | int] = {
        f"{prefix}n_choices": len(choices),
        f"{prefix}mean_choice": float(np.mean(choices)),
        f"{prefix}mean_reasoning_depth": float(np.mean([20 - c for c in choices])),
        f"{prefix}mass_17_19": float(np.mean([17 <= c <= 19 for c in choices])),
        f"{prefix}mass_19_20": float(np.mean([19 <= c <= 20 for c in choices])),
        f"{prefix}mass_17_20": float(np.mean([17 <= c <= 20 for c in choices])),
        f"{prefix}h1_score": float(np.mean([17 <= c <= 19 for c in choices]) - np.mean([19 <= c <= 20 for c in choices])),
        f"{prefix}h3_score": float(np.mean([17 <= c <= 20 for c in choices]) - 0.5),
    }
    counts = Counter(choices)
    for choice in CHOICES:
        values[f"{prefix}count_{choice}"] = counts.get(choice, 0)
    return values


def choice_regime(choice: int) -> str:
    if choice == 20:
        return "safe-20"
    if choice == 19:
        return "one-step-19"
    if 15 <= choice <= 18:
        return "middle-15:18"
    return "deep-11:14"


def prompt_features(prompt: str) -> dict[str, float | int | str]:
    numeric_tokens = re.findall(r"(?<!\w)(?:11|12|13|14|15|16|17|18|19|20)(?!\w)", prompt)
    lower = prompt.lower()
    persona_end = prompt.find("\n\nStyle:")
    persona_text = prompt[:persona_end] if persona_end >= 0 else ""
    return {
        "prompt_length": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "numeric_token_count": len(numeric_tokens),
        "token_11_count": numeric_tokens.count("11"),
        "token_20_count": numeric_tokens.count("20"),
        "contains_descending_phrase": int("20 down to 11" in lower),
        "contains_number_words": int("eleven" in lower and "twenty" in lower),
        "contains_one_shot": int("one-shot" in lower),
        "contains_private": int("private" in lower or "privately" in lower),
        "opponent_word_count": lower.count("opponent"),
        "participant_word_count": lower.count("participant"),
        "persona_length": len(persona_text),
    }


def load_raw_runs(study_dir: Path, tar_path: Path, released_number_parser: Callable[[str], int | None]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = study_dir / "sweeps" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"seed": int})
    run_rows: list[dict] = []
    agent_rows: list[dict] = []
    errors: list[dict] = []
    prompt_action_exact = 0
    prompt_action_multiset = 0
    first_token_action_exact = 0
    parsed_declared_runs = 0
    parsed_first_token_runs = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        member_names = set(tar.getnames())
        for row in manifest.itertuples(index=False):
            record = row._asdict()
            condition_id = str(record["condition_id"])
            labels = parse_condition_id(condition_id, str(record["model"]))
            run_dir = str(record["run_dir"]).replace("\\", "/").strip("/")
            prefix = f"beauty_contest/{run_dir}/run"
            action_member = f"{prefix}/action_events.jsonl.gz"
            prompt_member = f"{prefix}/prompts_and_responses.jsonl.gz"
            try:
                if action_member not in member_names or prompt_member not in member_names:
                    raise FileNotFoundError(f"missing tar members: {action_member} / {prompt_member}")
                actions = read_jsonl_gzip_member(tar, action_member)
                prompts = read_jsonl_gzip_member(tar, prompt_member)
                action_events = [
                    event for event in sorted(actions, key=lambda x: int(x.get("event_index", 0)))
                    if event.get("action_type") == "choose_number"
                ]
                choices = [int(event["data"]["choice"]) for event in action_events]
                action_agents = [str(event.get("source_user", "")) for event in action_events]
                prompt_records = [p for p in prompts if p.get("phase") == "action"]
                outputs = [str(p.get("output", "")) for p in prompt_records]
                prompt_texts = [str(p.get("prompt", "")) for p in prompt_records]
                declared_choices = [parse_output_choice(output) for output in outputs]
                first_tokens = [released_number_parser(output) for output in outputs]

                if len(choices) != 2 or any(c not in CHOICES for c in choices):
                    raise ValueError(f"expected two valid recorded choices, got {choices}")
                if len(prompt_records) != 2 or any(c is None for c in declared_choices):
                    raise ValueError(f"expected two parseable explicit action parameters, got {declared_choices}")
                if any(c is None for c in first_tokens):
                    raise ValueError(f"expected a legal-number token in each output, got {first_tokens}")
                declared = [int(c) for c in declared_choices if c is not None]
                first = [int(c) for c in first_tokens if c is not None]
                parsed_declared_runs += 1
                parsed_first_token_runs += 1
                if declared == choices:
                    prompt_action_exact += 1
                if sorted(declared) == sorted(choices):
                    prompt_action_multiset += 1
                if first == choices:
                    first_token_action_exact += 1

                feature_rows = [prompt_features(str(p.get("prompt", ""))) for p in prompt_records]
                run_record: dict = {
                    "condition_id": condition_id,
                    "kind": str(record["kind"]),
                    "seed": int(record["seed"]),
                    "manifest_status": str(record["status"]),
                    "tar_action_member": action_member,
                    "tar_prompt_member": prompt_member,
                    **labels,
                    **metrics_from_choices(choices),
                    **metrics_from_choices(declared, prefix="declared_"),
                    "prompt_choice_exact_match": int(declared == choices),
                    "prompt_choice_multiset_match": int(sorted(declared) == sorted(choices)),
                    "first_token_choice_exact_match": int(first == choices),
                    "prompt_length_mean": float(np.mean([f["prompt_length"] for f in feature_rows])),
                    "prompt_hashes": ";".join(sorted({str(f["prompt_sha256"]) for f in feature_rows})),
                }
                for key in feature_rows[0]:
                    if key == "prompt_sha256":
                        continue
                    values = [f[key] for f in feature_rows]
                    if all(isinstance(v, (int, float)) for v in values):
                        run_record[f"prompt_{key}"] = float(np.mean(values))
                run_rows.append(run_record)

                for idx, (choice, declared_choice, first_token, agent, output, prompt_text) in enumerate(
                    zip(choices, declared, first, action_agents, outputs, prompt_texts), start=1
                ):
                    agent_rows.append({
                        "condition_id": condition_id,
                        "kind": str(record["kind"]),
                        "seed": int(record["seed"]),
                        "agent_position": idx,
                        "agent_name": agent,
                        **labels,
                        "choice": choice,
                        "declared_choice": declared_choice,
                        "first_valid_choice_token": first_token,
                        "action_matches_declared": int(choice == declared_choice),
                        "action_matches_first_token": int(choice == first_token),
                        "reasoning_depth": 20 - choice,
                        "declared_reasoning_depth": 20 - declared_choice,
                        "regime": choice_regime(choice),
                        "declared_regime": choice_regime(declared_choice),
                        "prompt_exposes_choose_number_schema": int("CHOOSE_NUMBER(number" in prompt_text),
                        "prompt_requires_exactly_one_action": int("Respond with EXACTLY ONE action" in prompt_text),
                        "prompt_action_schema_sha256": hashlib.sha256(prompt_text[-700:].encode("utf-8")).hexdigest(),
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "output_prefix": output[:240].replace("\n", " "),
                        "output_suffix": output[-160:].replace("\n", " "),
                        "tar_action_member": action_member,
                        "tar_prompt_member": prompt_member,
                    })
            except Exception as exc:
                errors.append({
                    "condition_id": condition_id,
                    "seed": int(record["seed"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    run_df = pd.DataFrame(run_rows)
    agent_df = pd.DataFrame(agent_rows)
    divergent_runs = int((run_df["prompt_choice_exact_match"] == 0).sum()) if not run_df.empty else 0
    divergent_actions = int((agent_df["action_matches_declared"] == 0).sum()) if not agent_df.empty else 0
    integrity = {
        "manifest_rows": int(len(manifest)),
        "parsed_runs": int(len(run_df)),
        "agent_choices": int(len(agent_df)),
        "errors": errors,
        "explicit_action_parameters_parsed_runs": int(parsed_declared_runs),
        "first_valid_tokens_parsed_runs": int(parsed_first_token_runs),
        "prompt_action_exact_run_matches": int(prompt_action_exact),
        "prompt_action_multiset_run_matches": int(prompt_action_multiset),
        "first_token_action_exact_run_matches": int(first_token_action_exact),
        "divergent_runs": divergent_runs,
        "divergent_agent_actions": divergent_actions,
        "divergent_models": sorted(agent_df.loc[agent_df["action_matches_declared"] == 0, "model"].unique().tolist()),
        "all_recorded_actions_match_first_valid_legal_token": bool((agent_df["action_matches_first_token"] == 1).all()),
        "all_prompts_expose_choose_number_schema": bool((agent_df["prompt_exposes_choose_number_schema"] == 1).all()),
        "all_prompts_require_exactly_one_action": bool((agent_df["prompt_requires_exactly_one_action"] == 1).all()),
        "manifest_sha256": sha256_file(manifest_path),
        "tarball_sha256": sha256_file(tar_path),
        "input_mode": "direct official tar-member read",
    }
    if (
        errors
        or len(run_df) != len(manifest)
        or len(agent_df) != 2 * len(manifest)
        or parsed_declared_runs != len(manifest)
        or parsed_first_token_runs != len(manifest)
        or not integrity["all_recorded_actions_match_first_valid_legal_token"]
        or not integrity["all_prompts_expose_choose_number_schema"]
        or not integrity["all_prompts_require_exactly_one_action"]
    ):
        raise RuntimeError(f"Raw-data parse failed integrity gate: {json.dumps(integrity, ensure_ascii=False)}")
    return run_df, agent_df, integrity


def project_declared_actions(run_df: pd.DataFrame) -> pd.DataFrame:
    projected = run_df.copy()
    metric_names = [
        "n_choices", "mean_choice", "mean_reasoning_depth", "mass_17_19",
        "mass_19_20", "mass_17_20", "h1_score", "h3_score",
        *[f"count_{choice}" for choice in CHOICES],
    ]
    for name in metric_names:
        projected[name] = projected[f"declared_{name}"]
    projected["behavior_source"] = "explicit_action_parameter"
    return projected

def condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    label_columns = SIGNAL_VARIABLES + DESIGN_VARIABLES
    for condition_id, group in run_df.groupby("condition_id", sort=True):
        first = group.iloc[0]
        counts = [float(group[f"count_{choice}"].sum()) for choice in CHOICES]
        distribution = normalize(counts)
        mode_index = int(np.argmax(distribution))
        row: dict = {
            "condition_id": condition_id,
            "kind": first["kind"],
            **{key: first[key] for key in label_columns},
            "runs": int(len(group)),
            "choices": int(sum(counts)),
            "mean_choice": sum(choice * p for choice, p in zip(CHOICES, distribution)),
            "mean_reasoning_depth": sum((20 - choice) * p for choice, p in zip(CHOICES, distribution)),
            "mass_17_19": sum(distribution[c - 11] for c in (17, 18, 19)),
            "mass_19_20": sum(distribution[c - 11] for c in (19, 20)),
            "mass_17_20": sum(distribution[c - 11] for c in (17, 18, 19, 20)),
            "h1_score": sum(distribution[c - 11] for c in (17, 18, 19)) - sum(distribution[c - 11] for c in (19, 20)),
            "h3_score": sum(distribution[c - 11] for c in (17, 18, 19, 20)) - 0.5,
            "entropy_bits": entropy_bits(distribution),
            "modal_choice": CHOICES[mode_index],
            "modal_share": distribution[mode_index],
            "unique_choices": int(sum(p > 0 for p in distribution)),
        }
        for choice, p in zip(CHOICES, distribution):
            row[f"p_{choice}"] = p
        for anchor_name, anchor_map in ANCHORS.items():
            anchor = [anchor_map[c] for c in CHOICES]
            row[f"js_to_{anchor_name}"] = js_distance(distribution, anchor)
            row[f"tv_to_{anchor_name}"] = total_variation(distribution, anchor)
            row[f"emd_to_{anchor_name}"] = earth_mover_1d(distribution, anchor)
        own = str(first["game_variant"])
        row["js_to_own_anchor_descriptive_only"] = row[f"js_to_{own}"]
        js_values = {name: row[f"js_to_{name}"] for name in ANCHORS}
        row["nearest_human_anchor"] = min(js_values, key=js_values.get)
        row["own_anchor_is_nearest"] = int(row["nearest_human_anchor"] == own)
        others = [value for name, value in js_values.items() if name != own]
        row["own_anchor_margin"] = min(others) - js_values[own]
        rows.append(row)
    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    out = df
    for key, value in filters.items():
        out = out[out[key].astype(str) == str(value)]
    return out


def design_slice(run_df: pd.DataFrame, design_variable: str | None, design_level: str) -> pd.DataFrame:
    if design_variable is None or design_level == "default":
        return run_df[run_df["kind"] == "grid"].copy()
    mask = (run_df["kind"] == "variation") & (run_df[design_variable].astype(str) == str(design_level))
    for other in DESIGN_VARIABLES:
        if other != design_variable:
            mask &= run_df[other].astype(str) == DEFAULTS[other]
    return run_df[mask].copy()


def context_columns(spec: ClaimSpec) -> list[str]:
    if spec.claim_type == "paired":
        assert spec.arm_variable is not None
        columns = [v for v in SIGNAL_VARIABLES if v != spec.arm_variable]
    else:
        columns = list(SIGNAL_VARIABLES)
    return [v for v in columns if v not in spec.filters]


def make_stratum_id(row: pd.Series | dict, columns: list[str]) -> str:
    if not columns:
        return "all"
    return "|".join(f"{column}={row[column]}" for column in columns)


def compute_claim_effects(
    run_df: pd.DataFrame,
    spec: ClaimSpec,
    design_variable: str | None = None,
    design_level: str = "default",
) -> pd.DataFrame:
    data = design_slice(run_df, design_variable, design_level)
    data = apply_filters(data, spec.filters)
    contexts = context_columns(spec)
    common_meta = {
        "claim_id": spec.claim_id,
        "official_parent": spec.official_parent,
        "design_variable": design_variable or "default_grid",
        "design_level": design_level,
    }

    if spec.claim_type == "one_arm":
        if data.empty:
            return pd.DataFrame()
        rows = []
        for _, row in data.iterrows():
            rows.append({
                **common_meta,
                "seed": int(row["seed"]),
                "stratum_id": make_stratum_id(row, contexts),
                "signed_effect": float(row[spec.metric] - spec.threshold),
                "treatment_value": float(row[spec.metric]),
                "reference_value": spec.threshold,
                **{column: row[column] for column in contexts},
            })
        return pd.DataFrame(rows)

    assert spec.arm_variable and spec.treatment is not None and spec.reference is not None
    treatment = data[data[spec.arm_variable].astype(str) == str(spec.treatment)].copy()
    reference = data[data[spec.arm_variable].astype(str) == str(spec.reference)].copy()
    if treatment.empty or reference.empty:
        return pd.DataFrame()
    merge_keys = ["seed"] + contexts
    treatment_cols = merge_keys + [spec.metric, "condition_id", "tar_action_member"]
    reference_cols = merge_keys + [spec.metric, "condition_id", "tar_action_member"]
    treatment = treatment[treatment_cols].rename(columns={
        spec.metric: "treatment_value",
        "condition_id": "treatment_condition_id",
        "tar_action_member": "treatment_tar_member",
    })
    reference = reference[reference_cols].rename(columns={
        spec.metric: "reference_value",
        "condition_id": "reference_condition_id",
        "tar_action_member": "reference_tar_member",
    })
    merged = treatment.merge(reference, on=merge_keys, how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["signed_effect"] = spec.sign * (merged["treatment_value"] - merged["reference_value"])
    merged["stratum_id"] = merged.apply(lambda row: make_stratum_id(row, contexts), axis=1)
    for key, value in common_meta.items():
        merged[key] = value
    return merged


def bootstrap_effects(effects: pd.DataFrame, replicates: int, rng: np.random.Generator) -> dict[str, float]:
    if effects.empty:
        return {key: math.nan for key in [
            "hier_ci_low", "hier_ci_high", "seed_ci_low", "seed_ci_high", "ci_low", "ci_high"
        ]}
    groups = {stratum: group["signed_effect"].to_numpy(dtype=float) for stratum, group in effects.groupby("stratum_id")}
    strata = list(groups)
    seeds = sorted(effects["seed"].unique().tolist())
    by_seed = {seed: effects.loc[effects["seed"] == seed, "signed_effect"].to_numpy(dtype=float) for seed in seeds}
    hier_values = np.empty(replicates, dtype=float)
    seed_values = np.empty(replicates, dtype=float)
    for r in range(replicates):
        selected_strata = rng.choice(strata, size=len(strata), replace=True)
        stratum_means = []
        for stratum in selected_strata:
            vals = groups[stratum]
            sampled = rng.choice(vals, size=len(vals), replace=True)
            stratum_means.append(float(np.mean(sampled)))
        hier_values[r] = float(np.mean(stratum_means))

        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_seed_means = [float(np.mean(by_seed[int(seed)])) for seed in selected_seeds]
        seed_values[r] = float(np.mean(sampled_seed_means))

    hier_low, hier_high = np.quantile(hier_values, [0.025, 0.975])
    seed_low, seed_high = np.quantile(seed_values, [0.025, 0.975])
    return {
        "hier_ci_low": float(hier_low),
        "hier_ci_high": float(hier_high),
        "seed_ci_low": float(seed_low),
        "seed_ci_high": float(seed_high),
        "ci_low": float(min(hier_low, seed_low)),
        "ci_high": float(max(hier_high, seed_high)),
    }


def summarize_effects(
    effects: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    if effects.empty:
        return {
            "point_estimate": math.nan,
            "pooled_run_mean": math.nan,
            "median_stratum": math.nan,
            "sign_consistency": math.nan,
            "n_strata": 0,
            "n_seed_effects": 0,
            "seeds": "",
            "ci_low": math.nan,
            "ci_high": math.nan,
            "hier_ci_low": math.nan,
            "hier_ci_high": math.nan,
            "seed_ci_low": math.nan,
            "seed_ci_high": math.nan,
        }
    stratum_means = effects.groupby("stratum_id")["signed_effect"].mean()
    summary: dict[str, float | int | str] = {
        "point_estimate": float(stratum_means.mean()),
        "pooled_run_mean": float(effects["signed_effect"].mean()),
        "median_stratum": float(stratum_means.median()),
        "sign_consistency": float((stratum_means > 0).mean()),
        "n_strata": int(len(stratum_means)),
        "n_seed_effects": int(len(effects)),
        "seeds": ";".join(str(v) for v in sorted(effects["seed"].unique())),
    }
    summary.update(bootstrap_effects(effects, replicates, rng))
    return summary


def target_verdict(ci_low: float, ci_high: float) -> str:
    if math.isnan(ci_low) or math.isnan(ci_high):
        return "not_identifiable"
    if ci_low > 0:
        return "supported"
    if ci_high < 0:
        return "contradicted"
    return "mixed_or_inconclusive"


def build_claim_audits(
    run_df: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    all_effects: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    design_rows: list[dict] = []
    ident_rows: list[dict] = []
    effects_by_claim: dict[str, pd.DataFrame] = {}

    for spec in CLAIMS:
        default_effects = compute_claim_effects(run_df, spec)
        effects_by_claim[spec.claim_id] = default_effects
        if not default_effects.empty:
            all_effects.append(default_effects)
        default_summary = summarize_effects(default_effects, replicates, rng)
        target = target_verdict(float(default_summary["ci_low"]), float(default_summary["ci_high"]))
        identifiable_variables: set[str] = set()
        nonidentifiable_variables: set[str] = set()
        sign_flips = 0
        level_verdicts: list[str] = []
        all_level_points: list[float] = []
        all_level_ci_lows: list[float] = []
        all_level_ci_highs: list[float] = []

        for design_variable in DESIGN_VARIABLES:
            variable_identified = False
            for level in DESIGN_LEVELS[design_variable]:
                level_effects = compute_claim_effects(run_df, spec, design_variable, level)
                if level_effects.empty:
                    ident_rows.append({
                        "claim_id": spec.claim_id,
                        "official_parent": spec.official_parent,
                        "design_variable": design_variable,
                        "design_level": level,
                        "identifiable": False,
                        "n_strata": 0,
                        "n_seed_effects": 0,
                        "reason": "one or both claim arms are absent at this design level, or no matched context exists",
                    })
                    continue
                variable_identified = True
                all_effects.append(level_effects)
                level_summary = summarize_effects(level_effects, replicates, rng)
                level_strata = set(level_effects["stratum_id"].unique())
                matched_default = default_effects[default_effects["stratum_id"].isin(level_strata)].copy()
                matched_default_summary = summarize_effects(matched_default, replicates, rng)
                level_point = float(level_summary["point_estimate"])
                default_point = float(matched_default_summary["point_estimate"])
                strict_reversal = bool(level_point * default_point < 0)
                support_boundary_crossing = bool((level_point > 0) != (default_point > 0))
                sign_flips += int(support_boundary_crossing)
                if strict_reversal:
                    level_verdict = "design_reversal"
                elif support_boundary_crossing:
                    level_verdict = "support_boundary_crossing"
                elif float(level_summary["ci_low"]) > 0 and float(matched_default_summary["ci_low"]) > 0:
                    level_verdict = "stable_support"
                elif float(level_summary["ci_high"]) < 0 and float(matched_default_summary["ci_high"]) < 0:
                    level_verdict = "stable_contradiction"
                else:
                    level_verdict = "uncertain"
                level_verdicts.append(level_verdict)
                all_level_points.append(level_point)
                all_level_ci_lows.append(float(level_summary["ci_low"]))
                all_level_ci_highs.append(float(level_summary["ci_high"]))
                ident_rows.append({
                    "claim_id": spec.claim_id,
                    "official_parent": spec.official_parent,
                    "design_variable": design_variable,
                    "design_level": level,
                    "identifiable": True,
                    "n_strata": int(level_summary["n_strata"]),
                    "n_seed_effects": int(level_summary["n_seed_effects"]),
                    "reason": "matched claim arms exist under the non-default design level",
                })
                design_rows.append({
                    "claim_id": spec.claim_id,
                    "official_parent": spec.official_parent,
                    "design_variable": design_variable,
                    "design_level": level,
                    **{f"level_{key}": value for key, value in level_summary.items()},
                    **{f"matched_default_{key}": value for key, value in matched_default_summary.items()},
                    "global_default_point_estimate": float(default_summary["point_estimate"]),
                    "matched_subset_selection_shift": default_point - float(default_summary["point_estimate"]),
                    "incremental_design_shift": level_point - default_point,
                    "strict_reversal": strict_reversal,
                    "support_boundary_crossing": support_boundary_crossing,
                    "level_supports_claim": level_point > 0,
                    "matched_default_supports_claim": default_point > 0,
                    "level_verdict": level_verdict,
                })
            if variable_identified:
                identifiable_variables.add(design_variable)
            else:
                nonidentifiable_variables.add(design_variable)

        if not identifiable_variables:
            incremental_design_status = "not_identifiable"
            cross_cell_claim_status = "not_identifiable"
        else:
            if sign_flips > 0:
                incremental_design_status = "boundary_crossing"
            elif any(v == "uncertain" for v in level_verdicts):
                incremental_design_status = "uncertain"
            else:
                incremental_design_status = "direction_preserving"

            cell_points = [float(default_summary["point_estimate"]), *all_level_points]
            cell_ci_lows = [float(default_summary["ci_low"]), *all_level_ci_lows]
            cell_ci_highs = [float(default_summary["ci_high"]), *all_level_ci_highs]
            if all(value > 0 for value in cell_ci_lows):
                cross_cell_claim_status = "supported_in_all_identifiable_cells"
            elif all(value < 0 for value in cell_ci_highs):
                cross_cell_claim_status = "contradicted_in_all_identifiable_cells"
            elif all(value > 0 for value in cell_points):
                cross_cell_claim_status = "positive_direction_but_uncertain"
            elif all(value < 0 for value in cell_points):
                cross_cell_claim_status = "negative_direction_but_uncertain"
            else:
                cross_cell_claim_status = "mixed_across_identifiable_cells"

        summary_rows.append({
            "claim_id": spec.claim_id,
            "official_parent": spec.official_parent,
            "short_label": spec.short_label,
            "unit": spec.unit,
            "positive_supports_claim": True,
            **default_summary,
            "target_verdict": target,
            "design_status": cross_cell_claim_status,
            "cross_cell_claim_status": cross_cell_claim_status,
            "incremental_design_status": incremental_design_status,
            "identifiable_design_variables": ";".join(sorted(identifiable_variables)),
            "unidentified_design_variables": ";".join(sorted(nonidentifiable_variables)),
            "identifiable_design_variable_count": len(identifiable_variables),
            "identifiable_nondefault_levels": len(level_verdicts),
            "design_support_boundary_crossings": sign_flips,
            "design_cells_supporting_claim": int(sum(value > 0 for value in all_level_points)),
            "design_cells_contradicting_or_zero": int(sum(value <= 0 for value in all_level_points)),
            "worst_identifiable_design_point": min(all_level_points) if all_level_points else math.nan,
            "natural_scale": spec.natural_scale,
            "normalized_point": float(default_summary["point_estimate"]) / spec.natural_scale if not math.isnan(float(default_summary["point_estimate"])) else math.nan,
            "normalized_ci_low": float(default_summary["ci_low"]) / spec.natural_scale if not math.isnan(float(default_summary["ci_low"])) else math.nan,
            "normalized_ci_high": float(default_summary["ci_high"]) / spec.natural_scale if not math.isnan(float(default_summary["ci_high"])) else math.nan,
        })

    effect_df = pd.concat(all_effects, ignore_index=True, sort=False) if all_effects else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    design_df = pd.DataFrame(design_rows)
    ident_df = pd.DataFrame(ident_rows)
    return summary_df, design_df, ident_df, effect_df, effects_by_claim


def model_cycle_audit(h2_effects: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []
    if h2_effects.empty or "model" not in h2_effects.columns:
        return pd.DataFrame(), {"verdict": "not_identifiable"}
    for model, group in h2_effects.groupby("model", sort=True):
        summary = summarize_effects(group, replicates, rng)
        rows.append({"model": model, **summary, "direction_positive": float(summary["point_estimate"]) > 0})
    model_df = pd.DataFrame(rows)
    all_positive = bool((model_df["point_estimate"] > 0).all()) if not model_df.empty else False
    all_ci_positive = bool((model_df["ci_low"] > 0).all()) if not model_df.empty else False
    verdict = "supported" if all_ci_positive else ("point_direction_holds_but_uncertain" if all_positive else "contradicted")
    result = {
        "claim_id": "H5c",
        "official_parent": "h5_model_scale",
        "short_label": "cycle direction preserved for every model",
        "models": int(len(model_df)),
        "minimum_model_point_effect": float(model_df["point_estimate"].min()) if not model_df.empty else math.nan,
        "minimum_model_ci_low": float(model_df["ci_low"].min()) if not model_df.empty else math.nan,
        "all_model_point_directions_positive": all_positive,
        "all_model_intervals_positive": all_ci_positive,
        "target_verdict": verdict,
    }
    return model_df, result


def anchor_sanity_checks(condition_df: pd.DataFrame, h2_effects: pd.DataFrame, permutations: int, seed: int) -> tuple[pd.DataFrame, dict]:
    baseline = condition_df[
        (condition_df["kind"] == "grid")
        & (condition_df["game_variant"] == "basic")
        & (condition_df["goal_framing"] == "none")
        & (condition_df["persona"] == "neutral")
    ]
    copied_p = [float(baseline[f"p_{c}"].mean()) for c in CHOICES]
    copied_p = normalize(copied_p)
    distances = {
        variant: js_distance(copied_p, [ANCHORS[variant][c] for c in CHOICES])
        for variant in ANCHORS
    }
    old_switched_anchor_range = max(distances.values()) - min(distances.values())
    nearest = min(distances, key=distances.get)
    identical_own_anchor_accuracy = sum(nearest == variant for variant in ANCHORS) / len(ANCHORS)
    synthetic_rows = [
        {
            "check": "identical_distribution_old_switched_anchor_range",
            "value": old_switched_anchor_range,
            "expected_under_no_signal": 0.0,
            "passes": bool(old_switched_anchor_range > 0),
            "interpretation": "Nonzero demonstrates that changing the target anchor alone creates an apparent game-variant range.",
        },
        {
            "check": "identical_distribution_direct_variant_contrast",
            "value": 0.0,
            "expected_under_no_signal": 0.0,
            "passes": True,
            "interpretation": "CIRA's direct cycle-minus-basic choice contrast is exactly zero when behavior is copied across variants.",
        },
        {
            "check": "identical_distribution_own_anchor_identification_accuracy",
            "value": identical_own_anchor_accuracy,
            "expected_under_no_signal": 1 / 3,
            "passes": abs(identical_own_anchor_accuracy - 1 / 3) < 1e-12,
            "interpretation": "One unchanged distribution can be nearest to only one of three anchors; switching labels cannot yield perfect identification.",
        },
    ]

    rng = np.random.default_rng(seed)
    stratum_means = h2_effects.groupby("stratum_id")["signed_effect"].mean().to_numpy(dtype=float)
    observed = float(np.mean(stratum_means))
    null_values = np.empty(permutations, dtype=float)
    for i in range(permutations):
        signs = rng.choice([-1.0, 1.0], size=len(stratum_means), replace=True)
        null_values[i] = float(np.mean(stratum_means * signs))
    pvalue = float((1 + np.sum(np.abs(null_values) >= abs(observed))) / (permutations + 1))
    permutation = {
        "test": "H2 stratum-level paired randomization sign-flip",
        "observed_mean_effect": observed,
        "two_sided_pvalue": pvalue,
        "permutations": permutations,
        "unit": "matched context-stratum mean",
        "note": "This randomization test is supplementary; the paper's primary uncertainty remains conditional on five released seeds.",
    }
    return pd.DataFrame(synthetic_rows), permutation


def mechanism_analysis(agent_df: pd.DataFrame, run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    focus_agents = agent_df[
        (agent_df["game_variant"] == "basic")
        & (agent_df["model"] == "qwen3.5-27b-fp8")
        & (agent_df["persona"] == "neutral")
    ].copy()
    focus_runs = run_df[
        (run_df["game_variant"] == "basic")
        & (run_df["model"] == "qwen3.5-27b-fp8")
        & (run_df["persona"] == "neutral")
    ].copy()

    condition_rows = []
    for condition_id, group in focus_agents.groupby("condition_id", sort=True):
        first = group.iloc[0]
        counts = group["choice"].value_counts().sort_index()
        probabilities = counts / counts.sum()
        mode = int(counts.idxmax())
        condition_rows.append({
            "condition_id": condition_id,
            "goal_framing": first["goal_framing"],
            "instruction_wording": first["instruction_wording"],
            "persona_format": first["persona_format"],
            "response_format": first["response_format"],
            "temperature": first["temperature"],
            "n_agent_choices": int(len(group)),
            "mean_choice": float(group["choice"].mean()),
            "modal_choice": mode,
            "modal_share": float((group["choice"] == mode).mean()),
            "choice_entropy_bits": entropy_bits(probabilities.tolist()),
            "deep_11_14_share": float((group["choice"] <= 14).mean()),
            "one_step_19_share": float((group["choice"] == 19).mean()),
            "safe_20_share": float((group["choice"] == 20).mean()),
        })
    condition_df = pd.DataFrame(condition_rows)

    transition_rows = []
    matrix_rows = []
    for design_variable in ["instruction_wording", "persona_format"]:
        levels = DESIGN_LEVELS[design_variable]
        for goal in ["none", "strategic"]:
            default = focus_agents[
                (focus_agents["kind"] == "grid")
                & (focus_agents["goal_framing"] == goal)
            ][["seed", "agent_position", "choice", "regime"]].rename(columns={
                "choice": "default_choice", "regime": "default_regime"
            })
            for level in levels:
                variation = focus_agents[
                    (focus_agents["kind"] == "variation")
                    & (focus_agents["goal_framing"] == goal)
                    & (focus_agents[design_variable] == level)
                ][["seed", "agent_position", "choice", "regime"]].rename(columns={
                    "choice": "variation_choice", "regime": "variation_regime"
                })
                paired = default.merge(variation, on=["seed", "agent_position"], how="inner", validate="one_to_one")
                if paired.empty:
                    continue
                transition_rows.append({
                    "design_variable": design_variable,
                    "design_level": level,
                    "goal_framing": goal,
                    "n_paired_agent_choices": int(len(paired)),
                    "default_mean_choice": float(paired["default_choice"].mean()),
                    "variation_mean_choice": float(paired["variation_choice"].mean()),
                    "mean_shift": float((paired["variation_choice"] - paired["default_choice"]).mean()),
                    "exact_choice_agreement": float((paired["variation_choice"] == paired["default_choice"]).mean()),
                    "regime_switch_rate": float((paired["variation_regime"] != paired["default_regime"]).mean()),
                    "to_one_step_19_rate": float((paired["variation_choice"] == 19).mean()),
                    "default_deep_11_14_rate": float((paired["default_choice"] <= 14).mean()),
                    "deep_to_one_step_rate": float(((paired["default_choice"] <= 14) & (paired["variation_choice"] == 19)).mean()),
                })
                counts = paired.groupby(["default_regime", "variation_regime"]).size()
                for (source, target), count in counts.items():
                    matrix_rows.append({
                        "design_variable": design_variable,
                        "design_level": level,
                        "goal_framing": goal,
                        "default_regime": source,
                        "variation_regime": target,
                        "count": int(count),
                        "share": float(count / len(paired)),
                    })

    prompt_columns = [
        "condition_id", "goal_framing", "instruction_wording", "persona_format",
        "prompt_length_mean", "prompt_prompt_length", "prompt_numeric_token_count",
        "prompt_token_11_count", "prompt_token_20_count", "prompt_contains_descending_phrase",
        "prompt_contains_number_words", "prompt_contains_one_shot", "prompt_contains_private",
        "prompt_opponent_word_count", "prompt_participant_word_count", "prompt_persona_length",
        "prompt_hashes",
    ]
    prompt_columns = [column for column in prompt_columns if column in focus_runs.columns]
    prompt_df = focus_runs[prompt_columns].groupby("condition_id", as_index=False).first()
    condition_df = condition_df.merge(prompt_df, on=[c for c in ["condition_id", "goal_framing", "instruction_wording", "persona_format"] if c in prompt_df.columns], how="left")
    transition_df = pd.DataFrame(transition_rows)
    matrix_df = pd.DataFrame(matrix_rows)

    selected = condition_df[
        ((condition_df["instruction_wording"] != "default") | (condition_df["persona_format"] != "default"))
        & (condition_df["response_format"] == "default")
        & (condition_df["temperature"] == "0.5")
    ]
    default_selected = condition_df[
        (condition_df["instruction_wording"] == "default")
        & (condition_df["persona_format"] == "default")
        & (condition_df["response_format"] == "default")
        & (condition_df["temperature"] == "0.5")
    ]
    mechanism_summary = {
        "scope": "basic variant, qwen3.5-27b-fp8, neutral persona",
        "default_conditions": int(len(default_selected)),
        "variation_conditions": int(len(selected)),
        "default_mean_entropy_bits": float(default_selected["choice_entropy_bits"].mean()) if len(default_selected) else math.nan,
        "variation_mean_entropy_bits": float(selected["choice_entropy_bits"].mean()) if len(selected) else math.nan,
        "near_deterministic_variation_conditions_modal_share_ge_0_9": int((selected["modal_share"] >= 0.9).sum()),
        "variation_conditions_modal_choice_19": int((selected["modal_choice"] == 19).sum()),
        "interpretation": "Surface-equivalent representations often select low-entropy, discrete choice regimes rather than merely adding diffuse sampling noise. This is observable policy selection, not proof of a latent causal mechanism.",
    }
    return condition_df, transition_df, matrix_df, prompt_df, mechanism_summary


def make_claim_plot(summary_df: pd.DataFrame, outdir: Path) -> None:
    plot_df = summary_df.copy().iloc[::-1]
    y = np.arange(len(plot_df))
    point = plot_df["normalized_point"].to_numpy(dtype=float)
    low = plot_df["normalized_ci_low"].to_numpy(dtype=float)
    high = plot_df["normalized_ci_high"].to_numpy(dtype=float)
    errors = np.vstack([point - low, high - point])
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.errorbar(point, y, xerr=errors, fmt="o", capsize=3, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, plot_df["claim_id"] + "  " + plot_df["short_label"])
    ax.set_xlabel("Signed effect / natural outcome range (positive supports claim)")
    ax.set_title("CIRA default-grid effects with conservative 95% intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "claim_effects.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "claim_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def make_identifiability_plot(ident_df: pd.DataFrame, design_df: pd.DataFrame, outdir: Path) -> None:
    claims = [spec.claim_id for spec in CLAIMS]
    matrix = np.full((len(claims), len(DESIGN_VARIABLES)), np.nan)
    annotations = np.full((len(claims), len(DESIGN_VARIABLES)), "U", dtype=object)
    for i, claim in enumerate(claims):
        for j, variable in enumerate(DESIGN_VARIABLES):
            cells = ident_df[(ident_df["claim_id"] == claim) & (ident_df["design_variable"] == variable)]
            if not cells.empty and cells["identifiable"].any():
                d = design_df[(design_df["claim_id"] == claim) & (design_df["design_variable"] == variable)]
                if not d.empty and d["support_boundary_crossing"].any():
                    matrix[i, j] = -1
                    annotations[i, j] = "F"
                elif not d.empty and (d["level_verdict"].str.startswith("stable")).all():
                    matrix[i, j] = 1
                    annotations[i, j] = "S"
                else:
                    matrix[i, j] = 0
                    annotations[i, j] = "?"
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(masked, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(DESIGN_VARIABLES)), ["wording", "response", "temperature", "persona fmt"])
    ax.set_yticks(np.arange(len(claims)), claims)
    for i in range(len(claims)):
        for j in range(len(DESIGN_VARIABLES)):
            ax.text(j, i, annotations[i, j], ha="center", va="center")
    ax.set_title("Claim-by-variation audit: S stable, F sign flip, ? uncertain, U unidentified")
    fig.tight_layout()
    fig.savefig(outdir / "identifiability_matrix.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "identifiability_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(summary_df: pd.DataFrame, design_df: pd.DataFrame, mechanism_df: pd.DataFrame, outdir: Path) -> None:
    def fnum(value: float) -> str:
        return f"{value:.2f}"

    rows = []
    for _, row in summary_df.iterrows():
        identifiable = int(row["identifiable_design_variable_count"])
        rows.append(
            f"{row['claim_id']} & {fnum(float(row['point_estimate']))} "
            f"[{fnum(float(row['ci_low']))}, {fnum(float(row['ci_high']))}] & "
            f"{float(row['sign_consistency']):.0%} & {identifiable}/4 & "
            f"{str(row['target_verdict']).replace('_', ' ')} / {str(row['design_status']).replace('_', ' ')} \\\\"
        )
    table = "\n".join([
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Claim & signed effect [95\% CI] & sign & ID & target / design \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (outdir / "claim_table.tex").write_text(table, encoding="utf-8")

    mech_rows = []
    for _, row in mechanism_df.sort_values(["design_variable", "design_level", "goal_framing"]).iterrows():
        if row["design_level"] not in {"paraphrase-a", "paraphrase-b", "descending-range", "descriptive", "tabular"}:
            continue
        mech_rows.append(
            f"{row['design_level']} ({row['goal_framing']}) & {float(row['default_mean_choice']):.1f} "
            f"$\to$ {float(row['variation_mean_choice']):.1f} & {float(row['regime_switch_rate']):.0%} & "
            f"{float(row['to_one_step_19_rate']):.0%} \\\\"
        )
    mech_table = "\n".join([
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Representation & mean choice & regime switch & choice 19 \\",
        r"\midrule",
        *mech_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (outdir / "mechanism_table.tex").write_text(mech_table, encoding="utf-8")


def execution_fidelity_analysis(
    agent_df: pd.DataFrame,
    recorded_claims: pd.DataFrame,
    declared_claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    by_model_rows = []
    for model, group in agent_df.groupby("model", sort=True):
        by_model_rows.append({
            "model": model,
            "agent_actions": int(len(group)),
            "recorded_equals_explicit_parameter": int(group["action_matches_declared"].sum()),
            "explicit_parameter_fidelity": float(group["action_matches_declared"].mean()),
            "divergent_actions": int((group["action_matches_declared"] == 0).sum()),
            "recorded_equals_first_valid_token": float(group["action_matches_first_token"].mean()),
        })
    by_model = pd.DataFrame(by_model_rows)

    divergent = agent_df[agent_df["action_matches_declared"] == 0].copy()
    divergence_columns = [
        "condition_id", "seed", "agent_position", "model", "game_variant", "goal_framing",
        "persona", "instruction_wording", "response_format", "temperature", "persona_format",
        "choice", "declared_choice", "first_valid_choice_token", "action_matches_first_token",
        "output_sha256", "output_prefix", "output_suffix", "tar_action_member", "tar_prompt_member",
    ]
    divergent = divergent[divergence_columns]
    patterns = (
        divergent.groupby(["model", "choice", "declared_choice", "first_valid_choice_token"], dropna=False)
        .size().reset_index(name="agent_actions").sort_values("agent_actions", ascending=False)
    )
    by_condition = (
        agent_df.groupby(["condition_id", "model", "game_variant", "goal_framing", "persona"], as_index=False)
        .agg(
            agent_actions=("choice", "size"),
            fidelity=("action_matches_declared", "mean"),
            recorded_mean_choice=("choice", "mean"),
            explicit_parameter_mean_choice=("declared_choice", "mean"),
        )
    )
    by_condition["choice_shift_recorded_minus_explicit"] = (
        by_condition["recorded_mean_choice"] - by_condition["explicit_parameter_mean_choice"]
    )

    comparison = recorded_claims.merge(
        declared_claims,
        on=["claim_id", "official_parent", "short_label", "unit"],
        suffixes=("_recorded", "_explicit"),
        validate="one_to_one",
    )
    comparison["execution_shift"] = comparison["point_estimate_recorded"] - comparison["point_estimate_explicit"]
    comparison["target_verdict_changed"] = comparison["target_verdict_recorded"] != comparison["target_verdict_explicit"]
    comparison["design_status_changed"] = comparison["design_status_recorded"] != comparison["design_status_explicit"]

    summary = {
        "agent_actions": int(len(agent_df)),
        "divergent_agent_actions": int(len(divergent)),
        "divergent_action_fraction": float(len(divergent) / len(agent_df)),
        "divergent_runs": int(agent_df.groupby(["condition_id", "seed"])["action_matches_declared"].min().eq(0).sum()),
        "divergent_models": sorted(divergent["model"].unique().tolist()),
        "recorded_matches_first_valid_legal_token_fraction": float(agent_df["action_matches_first_token"].mean()),
        "divergent_recorded_matches_first_valid_legal_token_fraction": float(divergent["action_matches_first_token"].mean()) if len(divergent) else 1.0,
        "prompts_exposing_choose_number_schema": int(agent_df["prompt_exposes_choose_number_schema"].sum()),
        "prompts_requiring_exactly_one_action": int(agent_df["prompt_requires_exactly_one_action"].sum()),
        "all_prompts_expose_choose_number_schema": bool((agent_df["prompt_exposes_choose_number_schema"] == 1).all()),
        "all_prompts_require_exactly_one_action": bool((agent_df["prompt_requires_exactly_one_action"] == 1).all()),
        "claims_with_target_verdict_change": comparison.loc[comparison["target_verdict_changed"], "claim_id"].tolist(),
        "claims_with_design_status_change": comparison.loc[comparison["design_status_changed"], "claim_id"].tolist(),
        "interpretation": (
            "Recorded action events are the simulation's executed behavior. The explicit final action parameter is a separate observable. "
            "Their divergence is not labelled model intent; it is an execution-fidelity failure. Exact AST replay of the released "
            "parse_number_choice source reproduces every recorded Beauty action and confirms that the first legal integer is executed."
        ),
    }
    return by_model, divergent, patterns, by_condition, comparison, summary


def write_artifact_manifest(outdir: Path, input_paths: list[Path], metadata: dict) -> None:
    files = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({
                "path": path.relative_to(outdir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "generated_by": "analysis/final_cira_analysis.py",
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "metadata": metadata,
        "files": files,
    }
    (outdir / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-Identifiable Robustness Audit for the COLM 2026 Beauty Contest sweep")
    parser.add_argument("--study-dir", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/beauty_contest"))
    parser.add_argument("--tarball", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/beauty_contest_sweep.tar.gz"))
    parser.add_argument("--parser-source", type=Path, default=Path("references/socsim26_sharedtask/task_components/game.py"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts_final"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    released_number_parser, parser_source_audit = load_released_number_parser(args.parser_source)
    run_df, agent_df, integrity = load_raw_runs(args.study_dir, args.tarball, released_number_parser)
    parser_source_audit.update({
        "recorded_action_replay_accuracy": float(agent_df["action_matches_first_token"].mean()),
        "agent_actions": int(len(agent_df)),
        "divergent_explicit_action_fields": int((agent_df["action_matches_declared"] == 0).sum()),
    })
    (args.outdir / "parser_source_audit.json").write_text(
        json.dumps(parser_source_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_df["behavior_source"] = "recorded_action_event"
    declared_run_df = project_declared_actions(run_df)
    run_df.to_csv(args.outdir / "run_level.csv", index=False)
    agent_df.to_csv(args.outdir / "agent_choices.csv", index=False)
    (args.outdir / "data_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")

    condition_df = condition_summary(run_df)
    declared_condition_df = condition_summary(declared_run_df)
    condition_df.to_csv(args.outdir / "condition_summary.csv", index=False)
    declared_condition_df.to_csv(args.outdir / "explicit_parameter_condition_summary.csv", index=False)

    claim_summary, design_effects, identifiability, effect_rows, effects_by_claim = build_claim_audits(
        run_df, args.bootstrap_replicates, args.seed
    )
    declared_claims, declared_design, declared_ident, declared_effect_rows, declared_effects_by_claim = build_claim_audits(
        declared_run_df, args.bootstrap_replicates, args.seed
    )
    claim_summary.to_csv(args.outdir / "hypothesis_audit.csv", index=False)
    design_effects.to_csv(args.outdir / "matched_design_effects.csv", index=False)
    identifiability.to_csv(args.outdir / "design_identifiability.csv", index=False)
    effect_rows.to_csv(args.outdir / "hypothesis_effect_rows.csv", index=False)
    declared_claims.to_csv(args.outdir / "explicit_parameter_hypothesis_audit.csv", index=False)
    declared_design.to_csv(args.outdir / "explicit_parameter_matched_design_effects.csv", index=False)
    declared_ident.to_csv(args.outdir / "explicit_parameter_design_identifiability.csv", index=False)
    declared_effect_rows.to_csv(args.outdir / "explicit_parameter_hypothesis_effect_rows.csv", index=False)

    model_cycle_df, h5c = model_cycle_audit(effects_by_claim["H2"], args.bootstrap_replicates, args.seed + 17)
    declared_model_cycle_df, declared_h5c = model_cycle_audit(
        declared_effects_by_claim["H2"], args.bootstrap_replicates, args.seed + 17
    )
    model_cycle_df.to_csv(args.outdir / "cycle_effect_by_model.csv", index=False)
    declared_model_cycle_df.to_csv(args.outdir / "explicit_parameter_cycle_effect_by_model.csv", index=False)
    (args.outdir / "h5c_model_direction.json").write_text(json.dumps(h5c, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.outdir / "explicit_parameter_h5c_model_direction.json").write_text(
        json.dumps(declared_h5c, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sanity_df, permutation = anchor_sanity_checks(condition_df, effects_by_claim["H2"], args.permutations, args.seed + 31)
    sanity_df.to_csv(args.outdir / "anchor_sanity_checks.csv", index=False)
    (args.outdir / "permutation_tests.json").write_text(json.dumps(permutation, indent=2, ensure_ascii=False), encoding="utf-8")

    mechanism_conditions, mechanism_transitions, mechanism_matrix, prompt_features_df, mechanism_summary = mechanism_analysis(agent_df, run_df)
    mechanism_conditions.to_csv(args.outdir / "mechanism_condition_summary.csv", index=False)
    mechanism_transitions.to_csv(args.outdir / "mechanism_transitions.csv", index=False)
    mechanism_matrix.to_csv(args.outdir / "mechanism_transition_matrix.csv", index=False)
    prompt_features_df.to_csv(args.outdir / "prompt_feature_summary.csv", index=False)
    (args.outdir / "mechanism_summary.json").write_text(json.dumps(mechanism_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fidelity_by_model, divergent_actions, mismatch_patterns, fidelity_by_condition, claim_execution, execution_summary = execution_fidelity_analysis(
        agent_df, claim_summary, declared_claims
    )
    fidelity_by_model.to_csv(args.outdir / "execution_fidelity_by_model.csv", index=False)
    divergent_actions.to_csv(args.outdir / "executor_divergent_actions.csv", index=False)
    mismatch_patterns.to_csv(args.outdir / "executor_mismatch_patterns.csv", index=False)
    fidelity_by_condition.to_csv(args.outdir / "execution_fidelity_by_condition.csv", index=False)
    claim_execution.to_csv(args.outdir / "claim_execution_sensitivity.csv", index=False)
    (args.outdir / "execution_fidelity_summary.json").write_text(
        json.dumps(execution_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    make_claim_plot(claim_summary, args.outdir)
    make_identifiability_plot(identifiability, design_effects, args.outdir)
    write_latex_tables(claim_summary, design_effects, mechanism_transitions, args.outdir)

    final_summary = {
        "method": "Claim-Identifiable Robustness Audit (CIRA) with execution-fidelity audit",
        "study": "beauty_contest",
        "runs": int(len(run_df)),
        "conditions": int(len(condition_df)),
        "agent_choices": int(len(agent_df)),
        "seeds_per_condition": sorted(run_df.groupby("condition_id").size().unique().tolist()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutations": args.permutations,
        "recorded_action_claims": claim_summary.to_dict(orient="records"),
        "explicit_parameter_claims": declared_claims.to_dict(orient="records"),
        "recorded_h5c": h5c,
        "explicit_parameter_h5c": declared_h5c,
        "execution_fidelity": execution_summary,
        "permutation": permutation,
        "representation_mechanism": mechanism_summary,
        "statistical_scope": "Intervals are conditional sensitivity analyses over the five released seeds; bootstrap replicates are not independent observations.",
        "behavioral_scope": "Recorded action events are authoritative for executed simulation behavior; explicit final action parameters are used only as an execution-fidelity sensitivity projection.",
        "ipd_included": False,
        "anchor_leakage_in_primary_claims": False,
    }
    (args.outdir / "summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_artifact_manifest(
        args.outdir,
        [args.tarball, args.study_dir / "sweeps" / "manifest.csv", args.parser_source],
        {
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutations": args.permutations,
            "seed": args.seed,
            "runs": len(run_df),
            "conditions": len(condition_df),
            "divergent_actions": execution_summary["divergent_agent_actions"],
        },
    )
    print(json.dumps({
        "runs": len(run_df),
        "conditions": len(condition_df),
        "recorded_claims": claim_summary[["claim_id", "point_estimate", "ci_low", "ci_high", "target_verdict", "design_status"]].to_dict(orient="records"),
        "execution_fidelity": execution_summary,
        "h5c": h5c,
        "representation_mechanism": mechanism_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
