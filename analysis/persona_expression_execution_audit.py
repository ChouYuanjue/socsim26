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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from final_cira_analysis import sha256_file, summarize_effects, target_verdict

SIGNALS = ["persona_richness", "action_prompt", "model", "interaction_pathway"]
DESIGNS = ["persona_paraphrase", "persona_order"]
DEFAULTS = {
    "interaction_pathway": "closed",
    "persona_paraphrase": "original",
    "persona_order": "original",
}
DESIGN_LEVELS = {"persona_paraphrase": ["reworded"], "persona_order": ["reversed"]}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    parent: str
    label: str
    arm: str
    treatment: str
    reference: str
    sign: float = 1.0


CLAIMS = [
    Claim("E1a", "h1_persona_bottleneck", "demographic exceeds generic diversity", "persona_richness", "demographic", "generic"),
    Claim("E1b", "h1_persona_bottleneck", "full biography exceeds demographic diversity", "persona_richness", "full-bio", "demographic"),
    Claim("E1c", "h1_persona_bottleneck", "sociopsychological exceeds full-biography diversity", "persona_richness", "sociopsychological", "full-bio"),
    Claim("E3a", "h3_model_monoculture", "4B diversity is at least 9B diversity", "model", "qwen3.5-4b", "qwen3.5-9b"),
    Claim("E3b", "h3_model_monoculture", "9B diversity is at least 27B diversity", "model", "qwen3.5-9b", "qwen3.5-27b-fp8"),
]


def read_jsonl(tar: tarfile.TarFile, member: str) -> list[dict]:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    raw = handle.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as z:
        text = z.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def labels_from_condition(condition_id: str, model: str) -> dict[str, str]:
    labels = {
        "persona_richness": "",
        "action_prompt": "",
        "model": model,
        **DEFAULTS,
    }
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in labels:
            labels[key] = value
    if not labels["persona_richness"] or not labels["action_prompt"]:
        raise ValueError(f"cannot parse condition {condition_id}")
    return labels


def parse_create_tweet_calls(output: str) -> list[str]:
    text = str(output or "").strip()
    if not text.startswith("tool_calls:"):
        return []
    expression = text[len("tool_calls:") :]
    try:
        body = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return []
    calls = list(body.elts) if isinstance(body, (ast.Tuple, ast.List)) else [body]
    statuses: list[str] = []
    for call in calls:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return []
        if call.func.id != "create_tweet" or not call.args:
            return []
        try:
            payload = ast.literal_eval(call.args[0])
        except Exception:
            return []
        if not isinstance(payload, dict) or "status" not in payload:
            return []
        statuses.append(str(payload["status"]))
    return statuses


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9']+", str(text).lower()))


def mean_pairwise_jaccard_distance(texts: list[str]) -> float:
    sets = [token_set(text) for text in texts]
    n = len(sets)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        left = sets[i]
        for j in range(i + 1, n):
            right = sets[j]
            union = len(left | right)
            similarity = len(left & right) / union if union else 1.0
            total += 1.0 - similarity
            pairs += 1
    return total / pairs


def text_metrics(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "jaccard_diversity": math.nan,
            "unique_text_ratio": math.nan,
            "mean_word_count": math.nan,
            "duplicate_text_ratio": math.nan,
        }
    normalized = [" ".join(text.lower().split()) for text in texts]
    return {
        "jaccard_diversity": mean_pairwise_jaccard_distance(texts),
        "unique_text_ratio": len(set(normalized)) / len(normalized),
        "mean_word_count": float(np.mean([len(re.findall(r"[A-Za-z0-9']+", text)) for text in texts])),
        "duplicate_text_ratio": 1.0 - len(set(normalized)) / len(normalized),
    }


MULTI_CALL_GUIDANCE = "You are allowed to output multiple tool calls"


def load_runs(study_dir: Path, tar_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = study_dir / "sweeps" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"seed": int})
    run_rows = []
    decision_rows = []
    errors = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = set(tar.getnames())
        for raw in manifest.itertuples(index=False):
            row = raw._asdict()
            condition_id = str(row["condition_id"])
            labels = labels_from_condition(condition_id, str(row["model"]))
            run_dir = str(row["run_dir"]).replace("\\", "/").strip("/")
            prefix = f"persona_expression/{run_dir}/run"
            action_member = f"{prefix}/action_events.jsonl.gz"
            prompt_member = f"{prefix}/prompts_and_responses.jsonl.gz"
            try:
                if action_member not in members or prompt_member not in members:
                    raise FileNotFoundError(f"missing {action_member} or {prompt_member}")
                events = [event for event in read_jsonl(tar, action_member) if event.get("label") == "post"]
                prompts = [record for record in read_jsonl(tar, prompt_member) if record.get("phase") == "action"]
                flattened: list[str] = []
                strict_first: list[str] = []
                multi_outputs = 0
                call_counts: list[int] = []
                for index, prompt in enumerate(prompts):
                    output = str(prompt.get("output", ""))
                    statuses = parse_create_tweet_calls(output)
                    if not statuses:
                        raise ValueError(f"unparsed output at decision {index}: {output[:200]}")
                    if MULTI_CALL_GUIDANCE not in str(prompt.get("prompt", "")):
                        raise ValueError("effective multi-tool-call guidance absent")
                    flattened.extend(statuses)
                    strict_first.append(statuses[0])
                    multi_outputs += int(len(statuses) > 1)
                    call_counts.append(len(statuses))
                    decision_rows.append({
                        "condition_id": condition_id,
                        "kind": str(row["kind"]),
                        "seed": int(row["seed"]),
                        "decision_index": index,
                        "episode": int(prompt.get("episode_idx", -1)),
                        "logged_agent_name": str(prompt.get("agent_name", "")),
                        **labels,
                        "tool_calls_in_output": len(statuses),
                        "extra_actions": len(statuses) - 1,
                        "first_status": statuses[0],
                        "all_statuses_json": json.dumps(statuses, ensure_ascii=False),
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "output_prefix": output[:300].replace("\n", " "),
                        "output_suffix": output[-220:].replace("\n", " "),
                        "tar_action_member": action_member,
                        "tar_prompt_member": prompt_member,
                    })
                event_texts = [str(event["data"]["post_text"]) for event in events]
                if event_texts != flattened:
                    raise ValueError("flattened tool-call texts do not exactly replay post events in sequence")
                executed_metrics = text_metrics(event_texts)
                strict_metrics = text_metrics(strict_first)
                run_rows.append({
                    "condition_id": condition_id,
                    "kind": str(row["kind"]),
                    "seed": int(row["seed"]),
                    "model": str(row["model"]),
                    **labels,
                    "model_decisions": len(prompts),
                    "executed_posts": len(events),
                    "extra_posts": len(events) - len(prompts),
                    "multi_action_outputs": multi_outputs,
                    "max_actions_in_one_output": max(call_counts),
                    **executed_metrics,
                    **{f"strict_{key}": value for key, value in strict_metrics.items()},
                    "tar_action_member": action_member,
                    "tar_prompt_member": prompt_member,
                })
            except Exception as exc:
                errors.append({"condition_id": condition_id, "seed": int(row["seed"]), "type": type(exc).__name__, "error": str(exc)})
    runs = pd.DataFrame(run_rows)
    decisions = pd.DataFrame(decision_rows)
    integrity = {
        "manifest_rows": int(len(manifest)),
        "parsed_runs": int(len(runs)),
        "model_decisions": int(len(decisions)),
        "executed_posts": int(runs["executed_posts"].sum()) if not runs.empty else 0,
        "all_event_posts_replayed_from_raw_tool_calls": not errors and len(runs) == len(manifest),
        "multi_call_guidance_present_for_every_decision": not errors and len(decisions) > 0,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "tarball_sha256": sha256_file(tar_path),
    }
    if errors or len(runs) != len(manifest):
        raise RuntimeError(json.dumps(integrity, indent=2, ensure_ascii=False))
    return runs, decisions, integrity


def signal_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["persona_paraphrase"] == "original") & (df["persona_order"] == "original")].copy()


def project_strict(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in ["jaccard_diversity", "unique_text_ratio", "mean_word_count", "duplicate_text_ratio"]:
        out[metric] = out[f"strict_{metric}"]
    return out


def claim_effects(df: pd.DataFrame, claim: Claim, design: str | None = None, level: str = "default") -> pd.DataFrame:
    if design is None or level == "default":
        data = signal_rows(df)
    else:
        mask = df[design].astype(str).eq(level)
        for other in DESIGNS:
            if other != design:
                mask &= df[other].astype(str).eq(DEFAULTS[other])
        data = df[mask].copy()
    contexts = [signal for signal in SIGNALS if signal != claim.arm]
    treatment = data[data[claim.arm] == claim.treatment]
    reference = data[data[claim.arm] == claim.reference]
    if treatment.empty or reference.empty:
        return pd.DataFrame()
    keys = ["seed"] + contexts
    t = treatment[keys + ["jaccard_diversity", "condition_id", "tar_action_member"]].rename(columns={
        "jaccard_diversity": "treatment_value", "condition_id": "treatment_condition", "tar_action_member": "treatment_tar_member"
    })
    r = reference[keys + ["jaccard_diversity", "condition_id", "tar_action_member"]].rename(columns={
        "jaccard_diversity": "reference_value", "condition_id": "reference_condition", "tar_action_member": "reference_tar_member"
    })
    merged = t.merge(r, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["signed_effect"] = claim.sign * (merged["treatment_value"] - merged["reference_value"])
    merged["stratum_id"] = merged.apply(lambda row: "|".join(f"{key}={row[key]}" for key in contexts), axis=1)
    merged["claim_id"] = claim.claim_id
    merged["official_parent"] = claim.parent
    merged["design_variable"] = design or "default_signal_rows"
    merged["design_level"] = level
    return merged


def audit_claims(df: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summaries = []
    design_rows = []
    effect_rows = []
    for claim in CLAIMS:
        default = claim_effects(df, claim)
        if not default.empty:
            effect_rows.append(default)
        default_summary = summarize_effects(default, replicates, rng)
        identifiable = []
        support = []
        for design, levels in DESIGN_LEVELS.items():
            for level in levels:
                effects = claim_effects(df, claim, design, level)
                if effects.empty:
                    design_rows.append({
                        "claim_id": claim.claim_id, "design_variable": design, "design_level": level,
                        "identifiable": False, "reason": "required claim arms are not both present under this design level"
                    })
                    continue
                effect_rows.append(effects)
                summary = summarize_effects(effects, replicates, rng)
                identifiable.append((design, level))
                support.append(float(summary["point_estimate"]) > 0)
                design_rows.append({
                    "claim_id": claim.claim_id, "design_variable": design, "design_level": level,
                    "identifiable": True, **summary,
                })
        if not identifiable:
            cross_status = "not_identifiable"
        elif all(support):
            cross_status = "positive_direction_all_identifiable_cells"
        elif not any(support):
            cross_status = "nonpositive_all_identifiable_cells"
        else:
            cross_status = "mixed_across_identifiable_cells"
        summaries.append({
            "claim_id": claim.claim_id,
            "official_parent": claim.parent,
            "short_label": claim.label,
            **default_summary,
            "target_verdict": target_verdict(float(default_summary["ci_low"]), float(default_summary["ci_high"])),
            "cross_design_status": cross_status,
            "identifiable_nondefault_levels": len(identifiable),
        })
    return pd.DataFrame(summaries), pd.DataFrame(design_rows), pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame()


def interaction_audit(df: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    data = signal_rows(df)
    contexts = ["persona_richness", "action_prompt", "model"]
    open_rows = data[data["interaction_pathway"] == "open"]
    closed_rows = data[data["interaction_pathway"] == "closed"]
    keys = ["seed"] + contexts
    merged = open_rows[keys + ["jaccard_diversity"]].merge(
        closed_rows[keys + ["jaccard_diversity"]], on=keys, suffixes=("_open", "_closed"), validate="one_to_one"
    )
    merged["signed_effect"] = merged["jaccard_diversity_open"] - merged["jaccard_diversity_closed"]
    merged["absolute_effect"] = merged["signed_effect"].abs()
    merged["stratum_id"] = merged.apply(lambda row: "|".join(f"{key}={row[key]}" for key in contexts), axis=1)
    rng = np.random.default_rng(seed)
    signed = summarize_effects(merged, replicates, rng)
    absolute = summarize_effects(merged.rename(columns={"signed_effect": "raw_signed", "absolute_effect": "signed_effect"}), replicates, rng)
    return pd.DataFrame([
        {"metric": "open_minus_closed", **signed, "direction": "signed"},
        {"metric": "absolute_open_closed_change", **absolute, "direction": "magnitude"},
    ])


def factor_ranges(df: pd.DataFrame) -> pd.DataFrame:
    grid = df[df["kind"] == "grid"].copy()
    rows = []
    for source in ("executed", "strict"):
        metric = "jaccard_diversity" if source == "executed" else "strict_jaccard_diversity"
        for factor in ("persona_richness", "action_prompt", "model"):
            means = grid.groupby(factor)[metric].mean()
            rows.append({
                "projection": source,
                "factor": factor,
                "minimum_level": str(means.idxmin()),
                "maximum_level": str(means.idxmax()),
                "minimum_mean": float(means.min()),
                "maximum_mean": float(means.max()),
                "range": float(means.max() - means.min()),
            })
    return pd.DataFrame(rows)


def write_manifest(outdir: Path, inputs: list[Path], metadata: dict) -> None:
    files = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({"path": path.relative_to(outdir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (outdir / "ARTIFACT_MANIFEST.json").write_text(json.dumps({
        "generated_by": "analysis/persona_expression_execution_audit.py",
        "inputs": [{"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs],
        "metadata": metadata,
        "files": files,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def source_measurement_unit_audit(
    readme_path: Path,
    world_path: Path,
    sim_path: Path,
    prompt_runtime_path: Path,
    resolver_runtime_path: Path,
    turn_policy_path: Path,
    wheel_path: Path,
) -> dict:
    readme = readme_path.read_text(encoding="utf-8")
    world = world_path.read_text(encoding="utf-8")
    sim = sim_path.read_text(encoding="utf-8")
    prompt_runtime = prompt_runtime_path.read_text(encoding="utf-8")
    resolver_runtime = resolver_runtime_path.read_text(encoding="utf-8")
    turn_policy = turn_policy_path.read_text(encoding="utf-8")
    checks = {
        "readme_claims_one_post_per_agent_per_step": "one post per agent per step" in readme,
        "world_sets_30_agents": "num_agents: 30" in world,
        "world_sets_4_steps": "num_steps: 4" in world,
        "scenario_uses_multi_tool_calling": "mode: multi" in sim,
        "scenario_uses_single_action_turn_policy": "built_in: single_action" in sim,
        "scenario_comment_calls_one_post_per_agent_step_clean_unit": "One post per agent per step" in sim,
        "runtime_multi_prompt_says_calls_execute_in_sequence": "actions will be executed" in prompt_runtime and "in sequence of calls" in prompt_runtime,
        "runtime_resolver_invokes_every_normalized_call": "for tool_name, payload in normalized_calls" in resolver_runtime,
        "runtime_single_action_policy_runs_one_agent_step": "Default policy: one action per active agent per step" in turn_policy,
    }
    if not all(checks.values()):
        raise RuntimeError(f"source measurement-unit audit failed: {checks}")
    return {
        "interpretation": (
            "The effective prompt intentionally permits multiple tool calls, and the runtime resolver executes all of them. "
            "The mismatch is therefore not model noncompliance: the study README and single_action policy define one post per "
            "agent-step as the measurement unit, while tool_calling.mode=multi expands one model decision into multiple post events."
        ),
        "checks": checks,
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in [readme_path, world_path, sim_path, prompt_runtime_path, resolver_runtime_path, turn_policy_path, wheel_path]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/persona_expression"))
    parser.add_argument("--tarball", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/persona_expression_sweep.tar.gz"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts_persona_expression_final"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--readme", type=Path, default=Path("references/socsim26_sharedtask/studies/persona_expression/README.md"))
    parser.add_argument("--world", type=Path, default=Path("references/socsim26_sharedtask/studies/persona_expression/scenario/world/default.yaml"))
    parser.add_argument("--sim-config", type=Path, default=Path("references/socsim26_sharedtask/studies/persona_expression/scenario/sim.yaml"))
    parser.add_argument("--runtime-action-prompts", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/runtime/prompts/action_prompts.py"))
    parser.add_argument("--runtime-resolver", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/environments/gm/components/resolve.py"))
    parser.add_argument("--runtime-turn-policy", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/simulation_engines/policies/turns.py"))
    parser.add_argument("--runtime-wheel", type=Path, default=Path("references/silisocs_0_2_0/silisocs-0.2.0-py3-none-any.whl"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    runs, decisions, integrity = load_runs(args.study_dir, args.tarball)
    runs.to_csv(args.outdir / "run_level.csv", index=False)
    decisions.to_csv(args.outdir / "decision_level.csv", index=False)
    decisions[decisions["extra_actions"] > 0].to_csv(args.outdir / "multi_action_outputs.csv", index=False)
    (args.outdir / "data_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    source_audit = source_measurement_unit_audit(
        args.readme, args.world, args.sim_config, args.runtime_action_prompts,
        args.runtime_resolver, args.runtime_turn_policy, args.runtime_wheel,
    )
    (args.outdir / "measurement_unit_source_audit.json").write_text(
        json.dumps(source_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    executed_summary, design_audit, effects = audit_claims(runs, args.bootstrap_replicates, args.seed)
    strict_runs = project_strict(runs)
    strict_summary, strict_design, strict_effects = audit_claims(strict_runs, args.bootstrap_replicates, args.seed)
    executed_summary.to_csv(args.outdir / "hypothesis_audit.csv", index=False)
    strict_summary.to_csv(args.outdir / "strict_one_action_hypothesis_audit.csv", index=False)
    design_audit.to_csv(args.outdir / "design_identifiability.csv", index=False)
    strict_design.to_csv(args.outdir / "strict_one_action_design_identifiability.csv", index=False)
    effects.to_csv(args.outdir / "hypothesis_effect_rows.csv", index=False)
    strict_effects.to_csv(args.outdir / "strict_one_action_hypothesis_effect_rows.csv", index=False)
    comparison = executed_summary.merge(strict_summary, on=["claim_id", "official_parent", "short_label"], suffixes=("_executed", "_strict"), validate="one_to_one")
    comparison["execution_shift"] = comparison["point_estimate_executed"] - comparison["point_estimate_strict"]
    comparison["target_verdict_changed"] = comparison["target_verdict_executed"] != comparison["target_verdict_strict"]
    comparison.to_csv(args.outdir / "claim_execution_sensitivity.csv", index=False)
    interaction_audit(runs, args.bootstrap_replicates, args.seed + 1).to_csv(args.outdir / "interaction_effect.csv", index=False)
    factor_ranges(runs).to_csv(args.outdir / "factor_ranges.csv", index=False)

    by_model = runs.groupby("model", as_index=False).agg(
        runs=("condition_id", "size"),
        decisions=("model_decisions", "sum"),
        executed_posts=("executed_posts", "sum"),
        extra_posts=("extra_posts", "sum"),
        expanded_runs=("extra_posts", lambda values: int((values > 0).sum())),
        mean_executed_diversity=("jaccard_diversity", "mean"),
        mean_strict_diversity=("strict_jaccard_diversity", "mean"),
    )
    by_model["extra_posts_per_decision"] = by_model["extra_posts"] / by_model["decisions"]
    by_model.to_csv(args.outdir / "execution_fidelity_by_model.csv", index=False)
    call_counts = decisions.groupby("tool_calls_in_output", as_index=False).size().rename(columns={"size": "model_decisions"})
    call_counts.to_csv(args.outdir / "actions_per_decision_distribution.csv", index=False)

    summary = {
        "study": "persona_expression",
        "runs": int(len(runs)),
        "model_decisions": int(len(decisions)),
        "executed_posts": int(runs["executed_posts"].sum()),
        "multi_action_outputs": int((decisions["tool_calls_in_output"] > 1).sum()),
        "extra_posts": int(runs["extra_posts"].sum()),
        "expanded_runs": int((runs["extra_posts"] > 0).sum()),
        "maximum_actions_in_one_decision": int(decisions["tool_calls_in_output"].max()),
        "event_replay_accuracy": 1.0,
        "multi_call_guidance_coverage": 1.0,
        "maximum_absolute_run_diversity_shift": float((runs["jaccard_diversity"] - runs["strict_jaccard_diversity"]).abs().max()),
        "mean_absolute_run_diversity_shift": float((runs["jaccard_diversity"] - runs["strict_jaccard_diversity"]).abs().mean()),
        "claims_with_target_verdict_change": comparison.loc[comparison["target_verdict_changed"], "claim_id"].tolist(),
        "interpretation": (
            "The effective prompt explicitly permits multiple tool calls and the runtime executes every create_tweet call. "
            "However, the study README, world, and single_action policy describe one post per agent-step as the measurement unit. "
            "A one-decision/one-post projection keeps only the first call to quantify sensitivity to this configuration/measurement-unit mismatch; "
            "it is not privileged ground truth."
        ),
        "source_measurement_unit_audit": source_audit,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(args.outdir, [
        args.tarball, args.study_dir / "sweeps" / "manifest.csv", args.readme, args.world,
        args.sim_config, args.runtime_action_prompts, args.runtime_resolver,
        args.runtime_turn_policy, args.runtime_wheel,
    ], {
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        **{key: summary[key] for key in ["runs", "model_decisions", "executed_posts", "extra_posts"]},
    })
    print(json.dumps({
        **summary,
        "claims": executed_summary[["claim_id", "point_estimate", "ci_low", "ci_high", "target_verdict"]].to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
