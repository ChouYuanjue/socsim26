from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from final_cira_analysis import sha256_file, summarize_effects, target_verdict

SIGNAL_VARIABLES = [
    "instruction_framing",
    "payoff_scale",
    "vignette_frame",
    "model",
    "persona_stance",
]
DESIGN_VARIABLES = ["persona_format", "prompt_wording", "choice_labels", "history_format"]
DEFAULTS = {
    "instruction_framing": "canonical",
    "payoff_scale": "lambda-1",
    "vignette_frame": "abstract",
    "persona_stance": "neutral",
    "persona_format": "plain",
    "prompt_wording": "default",
    "choice_labels": "cooperate-defect",
    "history_format": "lines",
}
DESIGN_LEVELS = {
    "persona_format": ["descriptive", "tabular"],
    "prompt_wording": ["paraphrase-a", "paraphrase-b", "defect-first"],
    "choice_labels": ["green-blue", "action-ab"],
    "history_format": ["table", "summary_counts"],
}


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    official_parent: str
    short_label: str
    unit: str
    claim_type: str
    metric: str
    filters: dict[str, str]
    arm_variable: str | None = None
    treatment: str | None = None
    reference: str | None = None
    sign: float = 1.0
    threshold: float = 0.0
    natural_scale: float = 1.0


CLAIMS = [
    ClaimSpec(
        "P1a", "h1_instruction_framing", "moralized raises cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "instruction_framing", "moralized", "canonical", 1.0,
    ),
    ClaimSpec(
        "P1b", "h1_instruction_framing", "risk lowers cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "instruction_framing", "canonical", "risk", 1.0,
    ),
    ClaimSpec(
        "P2a", "h2_payoff_stakes", "attenuated stakes lower cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "payoff_scale", "lambda-1", "lambda-0.1", 1.0,
    ),
    ClaimSpec(
        "P2b", "h2_payoff_stakes", "amplified stakes do not lower cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "payoff_scale", "lambda-10", "lambda-1", 1.0,
    ),
    ClaimSpec(
        "P3", "h3_fictional_frame", "business exceeds fictional cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "vignette_frame", "business", "fictional", 1.0,
    ),
    ClaimSpec(
        "P5", "h5_endgame_defection", "early cooperation exceeds final-round cooperation", "cooperation-rate difference",
        "one_arm", "endgame_decline", {}, threshold=0.0,
    ),
    ClaimSpec(
        "P6a", "h6_persona_stance", "cooperative stance raises cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "persona_stance", "cooperative", "neutral", 1.0,
    ),
    ClaimSpec(
        "P6b", "h6_persona_stance", "competitive stance lowers cooperation", "cooperation-rate difference",
        "paired", "cooperation_rate", {}, "persona_stance", "neutral", "competitive", 1.0,
    ),
    ClaimSpec(
        "P6c", "h6_persona_stance", "reciprocal stance increases previous-opponent matching", "match-rate difference",
        "paired", "reciprocity_score", {}, "persona_stance", "reciprocal", "neutral", 1.0,
    ),
]


def read_jsonl_gzip_member(tar: tarfile.TarFile, member: str) -> list[dict]:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    raw = handle.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as zipped:
        text = zipped.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_condition_id(condition_id: str, model: str) -> dict[str, str]:
    labels = {**DEFAULTS, "model": model}
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in labels:
            labels[key] = value
    missing = [key for key in SIGNAL_VARIABLES if not labels.get(key)]
    if missing:
        raise ValueError(f"condition {condition_id} missing labels {missing}")
    return labels


def displayed_labels(choice_label: str) -> tuple[str, str]:
    if choice_label == "green-blue":
        return "GREEN", "BLUE"
    if choice_label == "action-ab":
        return "ACTION_A", "ACTION_B"
    return "COOPERATE", "DEFECT"


def load_released_parser_functions(source_path: Path) -> tuple[Callable, Callable, dict]:
    """Compile only the two parser functions from the released source file.

    This avoids importing simulator dependencies while executing the exact AST
    shipped in task_components/game.py.
    """
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    selected = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"parse_number_choice", "parse_pd_choice"}
    ]
    if {node.name for node in selected} != {"parse_number_choice", "parse_pd_choice"}:
        raise RuntimeError("released parser definitions not found")
    namespace: dict = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    lines = source.splitlines()
    locations = {}
    for node in selected:
        locations[node.name] = {
            "start_line": int(node.lineno),
            "end_line": int(node.end_lineno or node.lineno),
            "source": "\n".join(lines[node.lineno - 1 : int(node.end_lineno or node.lineno)]),
        }
    audit = {
        "source_path": source_path.as_posix(),
        "sha256": sha256_file(source_path),
        "functions": locations,
        "execution_mode": "exact AST definitions compiled with only the standard-library re dependency",
    }
    return namespace["parse_number_choice"], namespace["parse_pd_choice"], audit


def parse_explicit_pd_parameter(output: str, cooperate_label: str, defect_label: str) -> str | None:
    values = re.findall(r"(?im)^\s*choice\s*:\s*([^\r\n]+?)\s*$", str(output or ""))
    if not values:
        return None
    value = values[-1].strip().upper()
    if value == cooperate_label.upper():
        return "COOPERATE"
    if value == defect_label.upper():
        return "DEFECT"
    return None


def cooperation_metrics(round_choices: dict[int, list[str]]) -> dict[str, float]:
    values = [choice for rnd in sorted(round_choices) for choice in round_choices[rnd]]
    if len(values) != 20:
        raise ValueError(f"expected 20 choices, got {len(values)}")
    cooperation = np.asarray([choice == "COOPERATE" for choice in values], dtype=float)
    by_round = {
        rnd: float(np.mean([choice == "COOPERATE" for choice in choices]))
        for rnd, choices in round_choices.items()
    }
    early = float(np.mean([by_round[rnd] for rnd in (0, 1, 2)]))
    late = float(np.mean([by_round[rnd] for rnd in (7, 8, 9)]))
    mutual = float(np.mean([
        len(round_choices[rnd]) == 2 and all(choice == "COOPERATE" for choice in round_choices[rnd])
        for rnd in range(10)
    ]))
    return {
        "cooperation_rate": float(cooperation.mean()),
        "first_round_cooperation": by_round[0],
        "early_cooperation": early,
        "late_cooperation": late,
        "endgame_decline": early - late,
        "mutual_cooperation_rate": mutual,
    }


def recorded_reciprocity(actions: list[dict]) -> float:
    by_round_agent: dict[tuple[int, str], str] = {}
    agents: set[str] = set()
    for event in actions:
        rnd = int(event["data"]["round"])
        agent = str(event.get("source_user", ""))
        agents.add(agent)
        by_round_agent[(rnd, agent)] = str(event["data"]["choice"]).upper()
    if len(agents) != 2:
        raise ValueError(f"expected two agents, got {agents}")
    ordered = sorted(agents)
    matches = []
    for rnd in range(1, 10):
        for agent in ordered:
            opponent = ordered[1] if agent == ordered[0] else ordered[0]
            matches.append(by_round_agent[(rnd, agent)] == by_round_agent[(rnd - 1, opponent)])
    return float(np.mean(matches))


def load_ipd_runs(
    study_dir: Path,
    tar_path: Path,
    released_pd_parser: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = study_dir / "sweeps" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"seed": int})
    run_rows: list[dict] = []
    action_rows: list[dict] = []
    errors: list[dict] = []
    exact_round_matches = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        members = set(tar.getnames())
        for raw_row in manifest.itertuples(index=False):
            record = raw_row._asdict()
            condition_id = str(record["condition_id"])
            labels = parse_condition_id(condition_id, str(record["model"]))
            cooperate_label, defect_label = displayed_labels(labels["choice_labels"])
            run_dir = str(record["run_dir"]).replace("\\", "/").strip("/")
            prefix = f"iterated_pd/{run_dir}/run"
            action_member = f"{prefix}/action_events.jsonl.gz"
            prompt_member = f"{prefix}/prompts_and_responses.jsonl.gz"
            try:
                if action_member not in members or prompt_member not in members:
                    raise FileNotFoundError(f"missing {action_member} or {prompt_member}")
                action_events = [
                    event for event in read_jsonl_gzip_member(tar, action_member)
                    if event.get("action_type") == "choose_pd_action"
                ]
                prompt_rows = [
                    row for row in read_jsonl_gzip_member(tar, prompt_member)
                    if row.get("phase") == "action"
                ]
                if len(action_events) != 20 or len(prompt_rows) != 20:
                    raise ValueError(f"expected 20 actions/prompts, got {len(action_events)}/{len(prompt_rows)}")

                event_by_round: dict[int, list[str]] = defaultdict(list)
                for event in action_events:
                    event_by_round[int(event["data"]["round"])].append(str(event["data"]["choice"]).upper())

                replay_by_round: dict[int, list[str]] = defaultdict(list)
                explicit_by_round: dict[int, list[str]] = defaultdict(list)
                run_divergent = 0
                for prompt_index, prompt_row in enumerate(prompt_rows):
                    output = str(prompt_row.get("output", ""))
                    episode = int(prompt_row.get("episode_idx", -1))
                    replay = released_pd_parser(output, cooperate_label, defect_label)
                    explicit = parse_explicit_pd_parameter(output, cooperate_label, defect_label)
                    if replay not in {"COOPERATE", "DEFECT"} or explicit not in {"COOPERATE", "DEFECT"}:
                        raise ValueError(f"unparsed output at episode {episode}: replay={replay}, explicit={explicit}")
                    replay_by_round[episode].append(replay)
                    explicit_by_round[episode].append(explicit)
                    divergent = int(replay != explicit)
                    run_divergent += divergent
                    action_rows.append({
                        "condition_id": condition_id,
                        "kind": str(record["kind"]),
                        "seed": int(record["seed"]),
                        "prompt_index": prompt_index,
                        "round": episode,
                        "logged_agent_name": str(prompt_row.get("agent_name", "")),
                        **labels,
                        "cooperate_label": cooperate_label,
                        "defect_label": defect_label,
                        "recorded_choice_replayed": replay,
                        "explicit_action_parameter": explicit,
                        "divergent": divergent,
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "output_prefix": output[:300].replace("\n", " "),
                        "output_suffix": output[-220:].replace("\n", " "),
                        "tar_action_member": action_member,
                        "tar_prompt_member": prompt_member,
                    })

                for rnd in range(10):
                    if sorted(event_by_round[rnd]) != sorted(replay_by_round[rnd]):
                        raise ValueError(
                            f"released parser replay differs from events at round {rnd}: "
                            f"events={event_by_round[rnd]}, replay={replay_by_round[rnd]}"
                        )
                    exact_round_matches += 1

                recorded_metrics = cooperation_metrics(replay_by_round)
                explicit_metrics = cooperation_metrics(explicit_by_round)
                reciprocity = recorded_reciprocity(action_events)
                run_rows.append({
                    "condition_id": condition_id,
                    "kind": str(record["kind"]),
                    "model": str(record["model"]),
                    "seed": int(record["seed"]),
                    "manifest_status": str(record["status"]),
                    **labels,
                    **recorded_metrics,
                    **{f"explicit_{key}": value for key, value in explicit_metrics.items()},
                    "reciprocity_score": reciprocity,
                    # Strict projection is identical for every non-divergent run. Divergent
                    # runs cannot be assigned to players when prompt logs say "not found".
                    "explicit_reciprocity_score": reciprocity if run_divergent == 0 else math.nan,
                    "divergent_actions": run_divergent,
                    "tar_action_member": action_member,
                    "tar_prompt_member": prompt_member,
                })
            except Exception as exc:
                errors.append({
                    "condition_id": condition_id,
                    "seed": int(record["seed"]),
                    "type": type(exc).__name__,
                    "error": str(exc),
                })

    runs = pd.DataFrame(run_rows)
    actions = pd.DataFrame(action_rows)
    integrity = {
        "manifest_rows": int(len(manifest)),
        "parsed_runs": int(len(runs)),
        "prompt_action_calls": int(len(actions)),
        "expected_prompt_action_calls": int(20 * len(manifest)),
        "released_parser_round_matches": int(exact_round_matches),
        "expected_round_matches": int(10 * len(manifest)),
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "tarball_sha256": sha256_file(tar_path),
        "input_mode": "direct official tar-member read",
    }
    if errors or len(runs) != len(manifest) or len(actions) != 20 * len(manifest):
        raise RuntimeError(json.dumps(integrity, indent=2, ensure_ascii=False))
    if exact_round_matches != 10 * len(manifest):
        raise RuntimeError("released parser replay failed action-event integrity gate")
    return runs, actions, integrity


def project_explicit(run_df: pd.DataFrame) -> pd.DataFrame:
    projected = run_df.copy()
    metric_names = [
        "cooperation_rate", "first_round_cooperation", "early_cooperation",
        "late_cooperation", "endgame_decline", "mutual_cooperation_rate",
        "reciprocity_score",
    ]
    for metric in metric_names:
        projected[metric] = projected[f"explicit_{metric}"]
    return projected


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
    columns = [v for v in SIGNAL_VARIABLES if v != spec.arm_variable] if spec.claim_type == "paired" else list(SIGNAL_VARIABLES)
    return [column for column in columns if column not in spec.filters]


def make_stratum_id(row: pd.Series | dict, columns: list[str]) -> str:
    return "all" if not columns else "|".join(f"{column}={row[column]}" for column in columns)


def compute_claim_effects(
    run_df: pd.DataFrame,
    spec: ClaimSpec,
    design_variable: str | None = None,
    design_level: str = "default",
) -> pd.DataFrame:
    data = apply_filters(design_slice(run_df, design_variable, design_level), spec.filters)
    contexts = context_columns(spec)
    common = {
        "claim_id": spec.claim_id,
        "official_parent": spec.official_parent,
        "design_variable": design_variable or "default_grid",
        "design_level": design_level,
    }
    if spec.claim_type == "one_arm":
        rows = []
        for _, row in data.dropna(subset=[spec.metric]).iterrows():
            rows.append({
                **common,
                "seed": int(row["seed"]),
                "stratum_id": make_stratum_id(row, contexts),
                "signed_effect": float(row[spec.metric] - spec.threshold),
                "treatment_value": float(row[spec.metric]),
                "reference_value": float(spec.threshold),
                **{column: row[column] for column in contexts},
            })
        return pd.DataFrame(rows)

    assert spec.arm_variable and spec.treatment is not None and spec.reference is not None
    treatment = data[data[spec.arm_variable].astype(str) == str(spec.treatment)].dropna(subset=[spec.metric]).copy()
    reference = data[data[spec.arm_variable].astype(str) == str(spec.reference)].dropna(subset=[spec.metric]).copy()
    if treatment.empty or reference.empty:
        return pd.DataFrame()
    merge_keys = ["seed"] + contexts
    t = treatment[merge_keys + [spec.metric, "condition_id", "tar_action_member"]].rename(columns={
        spec.metric: "treatment_value", "condition_id": "treatment_condition_id", "tar_action_member": "treatment_tar_member"
    })
    r = reference[merge_keys + [spec.metric, "condition_id", "tar_action_member"]].rename(columns={
        spec.metric: "reference_value", "condition_id": "reference_condition_id", "tar_action_member": "reference_tar_member"
    })
    merged = t.merge(r, on=merge_keys, how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["signed_effect"] = spec.sign * (merged["treatment_value"] - merged["reference_value"])
    merged["stratum_id"] = merged.apply(lambda row: make_stratum_id(row, contexts), axis=1)
    for key, value in common.items():
        merged[key] = value
    return merged


def build_claim_audits(run_df: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    summaries = []
    design_rows = []
    ident_rows = []
    all_effects = []
    effects_by_claim: dict[str, pd.DataFrame] = {}

    for spec in CLAIMS:
        default_effects = compute_claim_effects(run_df, spec)
        effects_by_claim[spec.claim_id] = default_effects
        if not default_effects.empty:
            all_effects.append(default_effects)
        default_summary = summarize_effects(default_effects, replicates, rng)
        target = target_verdict(float(default_summary["ci_low"]), float(default_summary["ci_high"]))
        identifiable_variables: set[str] = set()
        unidentified_variables: set[str] = set()
        level_points: list[float] = []
        level_verdicts: list[str] = []
        boundary_crossings = 0
        strict_reversals = 0
        supporting_cells = 0
        nonsupporting_cells = 0

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
                matched_summary = summarize_effects(matched_default, replicates, rng)
                level_point = float(level_summary["point_estimate"])
                default_point = float(matched_summary["point_estimate"])
                supports = level_point > 0
                matched_supports = default_point > 0
                strict_reversal = bool(level_point * default_point < 0)
                boundary_crossing = bool(supports != matched_supports)
                supporting_cells += int(supports)
                nonsupporting_cells += int(not supports)
                strict_reversals += int(strict_reversal)
                boundary_crossings += int(boundary_crossing)
                level_points.append(level_point)
                if strict_reversal:
                    level_verdict = "design_reversal"
                elif boundary_crossing:
                    level_verdict = "support_boundary_crossing"
                elif float(level_summary["ci_low"]) > 0 and float(matched_summary["ci_low"]) > 0:
                    level_verdict = "stable_support"
                elif float(level_summary["ci_high"]) < 0 and float(matched_summary["ci_high"]) < 0:
                    level_verdict = "stable_contradiction"
                else:
                    level_verdict = "uncertain"
                level_verdicts.append(level_verdict)
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
                    **{f"matched_default_{key}": value for key, value in matched_summary.items()},
                    "global_default_point_estimate": float(default_summary["point_estimate"]),
                    "matched_subset_selection_shift": default_point - float(default_summary["point_estimate"]),
                    "incremental_design_shift": level_point - default_point,
                    "strict_reversal": strict_reversal,
                    "support_boundary_crossing": boundary_crossing,
                    "level_supports_claim": supports,
                    "matched_default_supports_claim": matched_supports,
                    "level_verdict": level_verdict,
                })
            if variable_identified:
                identifiable_variables.add(design_variable)
            else:
                unidentified_variables.add(design_variable)

        n_levels = len(level_points)
        if n_levels == 0:
            cross_cell_status = "not_identifiable"
        elif supporting_cells == n_levels:
            cross_cell_status = "positive_direction_all_identifiable_cells"
        elif supporting_cells == 0:
            cross_cell_status = "nonpositive_all_identifiable_cells"
        else:
            cross_cell_status = "mixed_across_identifiable_cells"
        if n_levels == 0:
            incremental_status = "not_identifiable"
        elif strict_reversals:
            incremental_status = "reversal"
        elif boundary_crossings:
            incremental_status = "boundary_crossing"
        elif any(verdict == "uncertain" for verdict in level_verdicts):
            incremental_status = "uncertain"
        else:
            incremental_status = "stable"

        summaries.append({
            "claim_id": spec.claim_id,
            "official_parent": spec.official_parent,
            "short_label": spec.short_label,
            "unit": spec.unit,
            **default_summary,
            "target_verdict": target,
            "cross_cell_claim_status": cross_cell_status,
            "incremental_design_status": incremental_status,
            "identifiable_design_variables": ";".join(sorted(identifiable_variables)),
            "unidentified_design_variables": ";".join(sorted(unidentified_variables)),
            "identifiable_design_variable_count": len(identifiable_variables),
            "identifiable_nondefault_levels": n_levels,
            "design_support_boundary_crossings": boundary_crossings,
            "design_strict_reversals": strict_reversals,
            "design_cells_supporting_claim": supporting_cells,
            "design_cells_nonpositive": nonsupporting_cells,
            "worst_identifiable_design_point": min(level_points) if level_points else math.nan,
            "natural_scale": spec.natural_scale,
        })

    summary_df = pd.DataFrame(summaries)
    design_df = pd.DataFrame(design_rows)
    ident_df = pd.DataFrame(ident_rows)
    effects_df = pd.concat(all_effects, ignore_index=True, sort=False) if all_effects else pd.DataFrame()
    return summary_df, design_df, ident_df, effects_df, effects_by_claim


def model_and_framing_audit(run_df: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = run_df[run_df["kind"] == "grid"].copy()
    model_summary = (
        grid.groupby("model", as_index=False)
        .agg(
            cooperation_rate=("cooperation_rate", "mean"),
            mutual_cooperation_rate=("mutual_cooperation_rate", "mean"),
            endgame_decline=("endgame_decline", "mean"),
            runs=("condition_id", "size"),
        )
        .sort_values("cooperation_rate", ascending=False)
    )
    rng = np.random.default_rng(seed)
    rows = []
    for claim_id in ("P1a", "P1b"):
        spec = next(spec for spec in CLAIMS if spec.claim_id == claim_id)
        effects = compute_claim_effects(run_df, spec)
        for model, group in effects.groupby("model", sort=True):
            summary = summarize_effects(group, replicates, rng)
            rows.append({
                "claim_id": claim_id,
                "model": model,
                **summary,
                "target_verdict": target_verdict(float(summary["ci_low"]), float(summary["ci_high"])),
            })
    return model_summary, pd.DataFrame(rows)


def parser_synthetic_checks(number_parser: Callable, pd_parser: Callable) -> pd.DataFrame:
    cases = [
        {
            "check": "money_reasoning_mentions_bonus_before_action",
            "text": "The 20-point bonus makes 19 attractive.\nACTION: CHOOSE_NUMBER\nnumber: 19",
            "released": number_parser("The 20-point bonus makes 19 attractive.\nACTION: CHOOSE_NUMBER\nnumber: 19"),
            "strict": 19,
            "expected_divergence": True,
        },
        {
            "check": "pd_reasoning_mentions_defect_before_cooperate_action",
            "text": "I will avoid triggering DEFECT.\nACTION: CHOOSE_PD_ACTION\nchoice: COOPERATE",
            "released": pd_parser("I will avoid triggering DEFECT.\nACTION: CHOOSE_PD_ACTION\nchoice: COOPERATE"),
            "strict": "COOPERATE",
            "expected_divergence": True,
        },
        {
            "check": "pd_relabelled_reasoning_mentions_blue_before_green_action",
            "text": "I will avoid BLUE.\nACTION: CHOOSE_PD_ACTION\nchoice: GREEN",
            "released": pd_parser("I will avoid BLUE.\nACTION: CHOOSE_PD_ACTION\nchoice: GREEN", "GREEN", "BLUE"),
            "strict": "COOPERATE",
            "expected_divergence": True,
        },
        {
            "check": "exact_action_only",
            "text": "ACTION: CHOOSE_PD_ACTION\nchoice: DEFECT",
            "released": pd_parser("ACTION: CHOOSE_PD_ACTION\nchoice: DEFECT"),
            "strict": "DEFECT",
            "expected_divergence": False,
        },
    ]
    for case in cases:
        case["diverges"] = case["released"] != case["strict"]
        case["passes"] = case["diverges"] == case["expected_divergence"]
    return pd.DataFrame(cases)


def write_manifest(outdir: Path, inputs: list[Path], metadata: dict) -> None:
    files = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({
                "path": path.relative_to(outdir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "generated_by": "analysis/ipd_cira_analysis.py",
        "inputs": [{"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs],
        "metadata": metadata,
        "files": files,
    }
    (outdir / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-identifiable and execution-fidelity audit for iterated PD")
    parser.add_argument("--study-dir", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/iterated_pd"))
    parser.add_argument("--tarball", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/iterated_pd_sweep.tar.gz"))
    parser.add_argument("--parser-source", type=Path, default=Path("references/socsim26_sharedtask/task_components/game.py"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts_ipd_final"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    number_parser, pd_parser, parser_audit = load_released_parser_functions(args.parser_source)
    runs, actions, integrity = load_ipd_runs(args.study_dir, args.tarball, pd_parser)
    runs.to_csv(args.outdir / "run_level.csv", index=False)
    actions.to_csv(args.outdir / "action_projection.csv", index=False)
    divergent = actions[actions["divergent"] == 1].copy()
    divergent.to_csv(args.outdir / "divergent_actions.csv", index=False)
    (args.outdir / "data_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")

    parser_audit.update({
        "released_parser_replays_every_action_event_round": True,
        "ipd_action_calls": int(len(actions)),
        "ipd_divergent_action_fields": int(len(divergent)),
        "ipd_divergent_runs": int(runs["divergent_actions"].gt(0).sum()),
    })
    (args.outdir / "parser_source_audit.json").write_text(json.dumps(parser_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    checks = parser_synthetic_checks(number_parser, pd_parser)
    checks.to_csv(args.outdir / "synthetic_parser_checks.csv", index=False)
    if not checks["passes"].all():
        raise RuntimeError("synthetic parser checks failed")

    recorded_summary, recorded_design, recorded_ident, recorded_effects, recorded_by_claim = build_claim_audits(
        runs, args.bootstrap_replicates, args.seed
    )
    explicit_runs = project_explicit(runs)
    explicit_summary, explicit_design, explicit_ident, explicit_effects, explicit_by_claim = build_claim_audits(
        explicit_runs, args.bootstrap_replicates, args.seed
    )
    recorded_summary.to_csv(args.outdir / "hypothesis_audit.csv", index=False)
    recorded_design.to_csv(args.outdir / "matched_design_effects.csv", index=False)
    recorded_ident.to_csv(args.outdir / "design_identifiability.csv", index=False)
    recorded_effects.to_csv(args.outdir / "hypothesis_effect_rows.csv", index=False)
    explicit_summary.to_csv(args.outdir / "explicit_parameter_hypothesis_audit.csv", index=False)
    explicit_design.to_csv(args.outdir / "explicit_parameter_matched_design_effects.csv", index=False)
    explicit_ident.to_csv(args.outdir / "explicit_parameter_design_identifiability.csv", index=False)
    explicit_effects.to_csv(args.outdir / "explicit_parameter_hypothesis_effect_rows.csv", index=False)

    comparison = recorded_summary.merge(
        explicit_summary,
        on=["claim_id", "official_parent", "short_label", "unit"],
        suffixes=("_recorded", "_explicit"),
        validate="one_to_one",
    )
    comparison["execution_shift"] = comparison["point_estimate_recorded"] - comparison["point_estimate_explicit"]
    comparison["target_verdict_changed"] = comparison["target_verdict_recorded"] != comparison["target_verdict_explicit"]
    comparison["cross_cell_status_changed"] = comparison["cross_cell_claim_status_recorded"] != comparison["cross_cell_claim_status_explicit"]
    comparison.to_csv(args.outdir / "claim_execution_sensitivity.csv", index=False)

    model_summary, framing_by_model = model_and_framing_audit(runs, args.bootstrap_replicates, args.seed + 101)
    model_summary.to_csv(args.outdir / "model_cooperation_summary.csv", index=False)
    framing_by_model.to_csv(args.outdir / "framing_effect_by_model.csv", index=False)

    by_model = (
        actions.groupby("model", as_index=False)
        .agg(action_calls=("divergent", "size"), divergent_actions=("divergent", "sum"))
    )
    by_model["explicit_field_fidelity"] = 1.0 - by_model["divergent_actions"] / by_model["action_calls"]
    by_model.to_csv(args.outdir / "execution_fidelity_by_model.csv", index=False)
    by_round = actions.groupby("round", as_index=False).agg(action_calls=("divergent", "size"), divergent_actions=("divergent", "sum"))
    by_round["divergence_rate"] = by_round["divergent_actions"] / by_round["action_calls"]
    by_round.to_csv(args.outdir / "execution_fidelity_by_round.csv", index=False)
    by_condition = (
        runs.groupby(["instruction_framing", "payoff_scale", "vignette_frame", "model", "persona_stance"], as_index=False)
        .agg(runs=("condition_id", "size"), divergent_actions=("divergent_actions", "sum"))
    )
    by_condition.to_csv(args.outdir / "execution_fidelity_by_signal_context.csv", index=False)

    summary = {
        "method": "Claim-Identifiable Robustness Audit with released-source execution replay",
        "study": "iterated_pd",
        "runs": int(len(runs)),
        "conditions": int(runs["condition_id"].nunique()),
        "action_calls": int(len(actions)),
        "divergent_action_fields": int(len(divergent)),
        "divergent_runs": int(runs["divergent_actions"].gt(0).sum()),
        "divergent_models": sorted(divergent["model"].unique().tolist()),
        "divergent_choice_label_schemes": sorted(divergent["choice_labels"].unique().tolist()),
        "released_parser_round_replay_accuracy": 1.0,
        "recorded_claims": recorded_summary.to_dict(orient="records"),
        "explicit_parameter_claims": explicit_summary.to_dict(orient="records"),
        "claims_with_target_verdict_change": comparison.loc[comparison["target_verdict_changed"], "claim_id"].tolist(),
        "claims_with_cross_cell_status_change": comparison.loc[comparison["cross_cell_status_changed"], "claim_id"].tolist(),
        "statistical_scope": "Intervals are conditional sensitivity analyses over five released seeds per condition; bootstrap draws are not independent observations.",
        "reciprocity_projection_note": "Strict player-level reciprocity is unavailable for divergent runs because some prompt records omit the agent identity; P6c remains identifiable because all reciprocal and neutral runs are non-divergent.",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(
        args.outdir,
        [args.tarball, args.study_dir / "sweeps" / "manifest.csv", args.parser_source],
        {
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "runs": len(runs),
            "action_calls": len(actions),
            "divergent_action_fields": len(divergent),
        },
    )
    print(json.dumps({
        "runs": len(runs),
        "conditions": runs["condition_id"].nunique(),
        "action_calls": len(actions),
        "divergent_action_fields": len(divergent),
        "divergent_runs": int(runs["divergent_actions"].gt(0).sum()),
        "claims": recorded_summary[["claim_id", "point_estimate", "ci_low", "ci_high", "target_verdict", "cross_cell_claim_status"]].to_dict(orient="records"),
        "claims_with_target_verdict_change": summary["claims_with_target_verdict_change"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
