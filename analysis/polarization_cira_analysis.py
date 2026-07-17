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

import numpy as np
import pandas as pd

from final_cira_analysis import sha256_file, summarize_effects, target_verdict

SIGNALS = ["exposure_rule", "graph_file", "memory_architecture", "model", "topic"]
DESIGNS = ["topic_paraphrase", "persona_content", "prompt_paraphrase"]
DEFAULTS = {
    "memory_architecture": "window",
    "model": "qwen3.5-27b-fp8",
    "topic": "euthanasia",
    "topic_paraphrase": "original",
    "persona_content": "default",
    "prompt_paraphrase": "default",
}
DESIGN_LEVELS = {
    "topic_paraphrase": ["reworded"],
    "persona_content": ["paraphrase", "narrative-stance", "order-b"],
    "prompt_paraphrase": ["reworded"],
}
GRAPH_FILES = {
    "scale-free": "scale_free_n32.json",
    "small-world": "small_world_n32.json",
    "random": "random_n32.json",
}
PERSONA_FILES = {
    "default": "default.json",
    "paraphrase": "paraphrase.json",
    "narrative-stance": "narrative.json",
    "order-b": "order_b.json",
}
ACTION_LABELS = {
    "create_tweet": "post",
    "reply_to_tweet": "reply",
    "like_tweet": "like",
    "repost_tweet": "repost",
}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    parent: str
    label: str
    metric: str
    claim_type: str
    filters: dict[str, str]
    arm: str | None = None
    treatment: str | None = None
    reference: str | None = None
    threshold: float = 0.0


CLAIMS = [
    Claim("Z1", "h1_exposure_rule", "stance-similar exposure raises edge-alignment gain", "edge_alignment_gain", "paired", {}, "exposure_rule", "stance-sim", "chrono"),
    Claim("Z6", "h6_opposite_exposure", "stance-similar exceeds opposite-stance clustering", "edge_alignment_gain", "paired", {}, "exposure_rule", "stance-sim", "stance-opp"),
    Claim("Z2a", "h2_topology", "scale-free exceeds random under chronological exposure", "edge_alignment_gain", "paired", {"exposure_rule": "chrono"}, "graph_file", "scale-free", "random"),
    Claim("Z2b", "h2_topology", "small-world exceeds random under chronological exposure", "edge_alignment_gain", "paired", {"exposure_rule": "chrono"}, "graph_file", "small-world", "random"),
    Claim("Z2c", "h2_topology", "scale-free exceeds random under stance-similar exposure", "edge_alignment_gain", "paired", {"exposure_rule": "stance-sim"}, "graph_file", "scale-free", "random"),
    Claim("Z2d", "h2_topology", "small-world exceeds random under stance-similar exposure", "edge_alignment_gain", "paired", {"exposure_rule": "stance-sim"}, "graph_file", "small-world", "random"),
    Claim("Z3", "h3_memory_architecture", "no-memory increases opinion volatility", "opinion_volatility", "paired", {"exposure_rule": "chrono", "graph_file": "scale-free"}, "memory_architecture", "none", "window"),
    Claim("Z4", "h4_credulity", "credulous personas shift more than skeptical personas", "credulity_shift_gap", "one_arm", {}, threshold=0.0),
]


def read_jsonl(tar: tarfile.TarFile, member: str) -> list[dict]:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    raw = handle.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as zipped:
        text = zipped.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_labels(condition_id: str, model: str) -> dict[str, str]:
    labels = {key: "" for key in ["exposure_rule", "graph_file"]}
    labels.update(DEFAULTS)
    labels["model"] = model
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in labels:
            labels[key] = value
    if not labels["exposure_rule"] or not labels["graph_file"]:
        raise ValueError(f"cannot parse {condition_id}")
    return labels


def parse_tool_call_names(output: str) -> list[str] | None:
    text = str(output or "").strip()
    if not text.startswith("tool_calls:"):
        return None
    expression = text[len("tool_calls:") :]
    try:
        body = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return []
    calls = list(body.elts) if isinstance(body, (ast.Tuple, ast.List)) else [body]
    names: list[str] = []
    for call in calls:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return []
        names.append(call.func.id)
    return names


def derive_agent_name(prompt: str) -> str | None:
    match = re.match(r"Persona:\n([^\n]+?) is ", str(prompt or ""))
    return match.group(1) if match else None


def load_personas(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows, {str(row["name"]): row for row in rows}


def unique_edges(graph: dict[str, list[str]]) -> list[tuple[int, int]]:
    return sorted({tuple(sorted((int(source), int(target)))) for source, targets in graph.items() for target in targets if int(source) != int(target)})


def edge_alignment(values: list[float], edges: list[tuple[int, int]]) -> float:
    return float(np.mean([1.0 - abs(values[left] - values[right]) / 4.0 for left, right in edges]))


def run_opinion_metrics(
    probe_events: list[dict],
    graph_path: Path,
    persona_path: Path,
) -> dict[str, float | int | bool]:
    personas, by_name = load_personas(persona_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    edges = unique_edges(graph)
    names = [str(row["name"]) for row in personas]
    initial = [float(row["stance"]) for row in personas]
    ratings: dict[int, dict[str, float]] = defaultdict(dict)
    strict_probe_outputs = 0
    missing_probe_outputs = 0
    for event in probe_events:
        raw_value = event["data"].get("probe_return")
        raw_response = event["data"].get("raw_response")
        if raw_value is None:
            missing_probe_outputs += 1
            continue
        value = float(raw_value)
        raw = str(raw_response or "").strip()
        if re.fullmatch(r"[1-5]", raw) and float(raw) == value:
            strict_probe_outputs += 1
        ratings[int(event["episode"])][str(event["source_user"])] = value
    complete = all(name in ratings.get(episode, {}) for episode in range(1, 15) for name in names)
    base = {
        "strict_probe_outputs": strict_probe_outputs,
        "missing_probe_outputs": missing_probe_outputs,
        "probe_calls": len(probe_events),
        "complete_probe_trajectory": complete,
    }
    if not complete:
        return {
            **base,
            "initial_edge_alignment": math.nan,
            "final_edge_alignment": math.nan,
            "edge_alignment_gain": math.nan,
            "initial_opinion_variance": float(np.var(initial, ddof=1)),
            "final_opinion_variance": math.nan,
            "opinion_variance_change": math.nan,
            "opinion_volatility": math.nan,
            "mean_absolute_opinion_shift": math.nan,
            "credulous_mean_shift": math.nan,
            "skeptical_mean_shift": math.nan,
            "credulity_shift_gap": math.nan,
        }
    trajectories: dict[str, list[float]] = {}
    for name, start_value in zip(names, initial):
        trajectories[name] = [start_value] + [ratings[episode][name] for episode in range(1, 15)]
    final = [trajectories[name][-1] for name in names]
    initial_alignment = edge_alignment(initial, edges)
    final_alignment = edge_alignment(final, edges)
    volatility = float(np.mean([abs(values[index] - values[index - 1]) for values in trajectories.values() for index in range(1, len(values))]))
    absolute_shifts = {name: abs(values[-1] - values[0]) for name, values in trajectories.items()}
    credulous = [absolute_shifts[name] for name in names if str(by_name[name].get("credulity")) == "credulous"]
    skeptical = [absolute_shifts[name] for name in names if str(by_name[name].get("credulity")) == "skeptical"]
    return {
        **base,
        "initial_edge_alignment": initial_alignment,
        "final_edge_alignment": final_alignment,
        "edge_alignment_gain": final_alignment - initial_alignment,
        "initial_opinion_variance": float(np.var(initial, ddof=1)),
        "final_opinion_variance": float(np.var(final, ddof=1)),
        "opinion_variance_change": float(np.var(final, ddof=1) - np.var(initial, ddof=1)),
        "opinion_volatility": volatility,
        "mean_absolute_opinion_shift": float(np.mean(list(absolute_shifts.values()))),
        "credulous_mean_shift": float(np.mean(credulous)),
        "skeptical_mean_shift": float(np.mean(skeptical)),
        "credulity_shift_gap": float(np.mean(credulous) - np.mean(skeptical)),
    }

def source_cap_audit(readme: Path, sim_config: Path, prompt_runtime: Path, resolver_runtime: Path, turn_runtime: Path, wheel: Path) -> dict:
    texts = {path: path.read_text(encoding="utf-8") for path in [readme, sim_config, prompt_runtime, resolver_runtime, turn_runtime]}
    checks = {
        "readme_declares_up_to_three_tool_calls": bool(re.search(r"up\s+to\s+3\s+tool\s+calls", texts[readme], flags=re.I)),
        "sim_open_ended_max_actions_three": "max_actions: 3" in texts[sim_config],
        "sim_multi_tool_calling": "mode: multi" in texts[sim_config],
        "runtime_prompt_allows_batched_calls": "multiple tool calls" in texts[prompt_runtime] and "executed" in texts[prompt_runtime],
        "resolver_executes_every_call_in_batch": "for tool_name, payload in normalized_calls" in texts[resolver_runtime],
        "turn_policy_counts_batch_after_agent_step": "actions_used += max(1, _count_structured_actions" in texts[turn_runtime],
    }
    if not all(checks.values()):
        raise RuntimeError(f"cap source audit failed: {checks}")
    return {
        "checks": checks,
        "interpretation": "The open-ended cap is checked after a batched response is resolved. A single batch may therefore execute more than the advertised three calls before the loop terminates.",
        "files": [{"path": path.as_posix(), "sha256": sha256_file(path)} for path in [readme, sim_config, prompt_runtime, resolver_runtime, turn_runtime, wheel]],
    }


def load_runs(
    study_dir: Path,
    tar_path: Path,
    assets_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = study_dir / "sweeps" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"seed": int})
    run_rows: list[dict] = []
    decision_rows: list[dict] = []
    overshoot_rows: list[dict] = []
    errors: list[dict] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = set(tar.getnames())
        for raw in manifest.itertuples(index=False):
            row = raw._asdict()
            condition_id = str(row["condition_id"])
            labels = parse_labels(condition_id, str(row["model"]))
            run_dir = str(row["run_dir"]).replace("\\", "/").strip("/")
            prefix = f"polarization/{run_dir}/run"
            action_member = f"{prefix}/action_events.jsonl.gz"
            probe_member = f"{prefix}/probe_events.jsonl.gz"
            prompt_member = f"{prefix}/prompts_and_responses.jsonl.gz"
            try:
                if any(member not in members for member in [action_member, probe_member, prompt_member]):
                    raise FileNotFoundError(prefix)
                events = read_jsonl(tar, action_member)
                probe_events = [event for event in read_jsonl(tar, probe_member) if event.get("label") == "opinion_rating"]
                prompts = [record for record in read_jsonl(tar, prompt_member) if record.get("phase") == "action"]
                social_events = [event for event in events if event.get("label") in set(ACTION_LABELS.values())]
                mapped_event_labels = [event["label"] for event in social_events]
                event_turn_counts: Counter[tuple[str, int]] = Counter(
                    (str(event.get("source_user", "")), int(event.get("episode", -1))) for event in social_events
                )
                mapped_calls: list[str] = []
                turn_calls: Counter[tuple[str, int]] = Counter()
                tool_decisions = 0
                memory_calls = 0
                for index, prompt in enumerate(prompts):
                    names = parse_tool_call_names(str(prompt.get("output", "")))
                    if names is None:
                        memory_calls += 1
                        continue
                    if not names:
                        raise ValueError(f"unparsed tool-call output at record {index}")
                    agent = derive_agent_name(str(prompt.get("prompt", "")))
                    if agent is None:
                        raise ValueError(f"cannot derive agent at record {index}")
                    episode = int(prompt.get("episode_idx", -1))
                    tool_decisions += 1
                    turn_calls[(agent, episode)] += sum(name in ACTION_LABELS for name in names)
                    mapped_calls.extend([ACTION_LABELS[name] for name in names if name in ACTION_LABELS])
                    decision_rows.append({
                        "condition_id": condition_id,
                        "kind": str(row["kind"]),
                        "seed": int(row["seed"]),
                        "decision_index": index,
                        "agent_name": agent,
                        "episode": episode,
                        **labels,
                        "tool_calls_in_batch": len(names),
                        "nonterminal_actions_in_batch": sum(name in ACTION_LABELS for name in names),
                        "contains_finished": int("FINISHED" in names),
                        "call_names_json": json.dumps(names),
                        "output_sha256": hashlib.sha256(str(prompt.get("output", "")).encode("utf-8")).hexdigest(),
                        "tar_action_member": action_member,
                        "tar_prompt_member": prompt_member,
                    })
                generated_counts = Counter(mapped_calls)
                executed_counts = Counter(mapped_event_labels)
                unexecuted_generated = generated_counts - executed_counts
                unexpected_events = executed_counts - generated_counts
                unexecuted_generated_calls = int(sum(unexecuted_generated.values()))
                unexpected_event_calls = int(sum(unexpected_events.values()))
                exact_action_sequence_match = mapped_calls == mapped_event_labels
                generated_calls_above_cap = sum(max(0, count - 3) for count in turn_calls.values())
                generated_overshoot_turns = sum(count > 3 for count in turn_calls.values())
                executed_actions_above_cap = sum(max(0, count - 3) for count in event_turn_counts.values())
                executed_overshoot_turns = sum(count > 3 for count in event_turn_counts.values())
                for agent_episode in sorted(set(turn_calls) | set(event_turn_counts)):
                    generated_count = int(turn_calls.get(agent_episode, 0))
                    executed_count = int(event_turn_counts.get(agent_episode, 0))
                    if generated_count > 3 or executed_count > 3:
                        agent, episode = agent_episode
                        overshoot_rows.append({
                            "condition_id": condition_id,
                            "kind": str(row["kind"]),
                            "seed": int(row["seed"]),
                            "agent_name": agent,
                            "episode": episode,
                            **labels,
                            "generated_nonterminal_calls": generated_count,
                            "executed_social_events": executed_count,
                            "generated_calls_above_cap": max(0, generated_count - 3),
                            "executed_events_above_cap": max(0, executed_count - 3),
                            "tar_prompt_member": prompt_member,
                            "tar_action_member": action_member,
                        })
                graph_path = assets_dir / "graphs" / GRAPH_FILES[labels["graph_file"]]
                persona_path = assets_dir / "personas" / PERSONA_FILES[labels["persona_content"]]
                metrics = run_opinion_metrics(probe_events, graph_path, persona_path)
                run_rows.append({
                    "condition_id": condition_id,
                    "kind": str(row["kind"]),
                    "seed": int(row["seed"]),
                    **labels,
                    **metrics,
                    "prompt_records": len(prompts),
                    "tool_call_decisions": tool_decisions,
                    "memory_or_probe_prompt_records": memory_calls,
                    "generated_nonterminal_actions": len(mapped_calls),
                    "executed_nonterminal_actions": len(mapped_event_labels),
                    "unexecuted_generated_calls": unexecuted_generated_calls,
                    "unexpected_event_calls": unexpected_event_calls,
                    "exact_action_sequence_match": exact_action_sequence_match,
                    "unexecuted_call_labels_json": json.dumps(dict(unexecuted_generated), sort_keys=True),
                    "unexpected_event_labels_json": json.dumps(dict(unexpected_events), sort_keys=True),
                    "generated_overshoot_turns": generated_overshoot_turns,
                    "generated_calls_above_cap": generated_calls_above_cap,
                    "executed_overshoot_turns": executed_overshoot_turns,
                    "executed_events_above_cap": executed_actions_above_cap,
                    "maximum_generated_calls_in_agent_episode": max(turn_calls.values()),
                    "maximum_executed_events_in_agent_episode": max(event_turn_counts.values()),
                    "tar_action_member": action_member,
                    "tar_probe_member": probe_member,
                    "tar_prompt_member": prompt_member,
                })
            except Exception as exc:
                errors.append({"condition_id": condition_id, "seed": int(row["seed"]), "type": type(exc).__name__, "error": str(exc)})
    runs = pd.DataFrame(run_rows)
    decisions = pd.DataFrame(decision_rows)
    overshoots = pd.DataFrame(overshoot_rows)
    integrity = {
        "manifest_rows": int(len(manifest)),
        "parsed_runs": int(len(runs)),
        "tool_call_decisions": int(len(decisions)),
        "generated_overshoot_turns": int((overshoots["generated_calls_above_cap"] > 0).sum()) if len(overshoots) else 0,
        "generated_calls_above_cap": int(overshoots["generated_calls_above_cap"].sum()) if len(overshoots) else 0,
        "executed_overshoot_turns": int((overshoots["executed_events_above_cap"] > 0).sum()) if len(overshoots) else 0,
        "executed_events_above_cap": int(overshoots["executed_events_above_cap"].sum()) if len(overshoots) else 0,
        "strict_probe_fidelity_nonmissing": float(runs["strict_probe_outputs"].sum() / max(1, runs["probe_calls"].sum() - runs["missing_probe_outputs"].sum())) if len(runs) else math.nan,
        "missing_probe_outputs": int(runs["missing_probe_outputs"].sum()) if len(runs) else 0,
        "incomplete_probe_runs": int((~runs["complete_probe_trajectory"]).sum()) if len(runs) else 0,
        "exact_action_sequence_runs": int(runs["exact_action_sequence_match"].sum()) if len(runs) else 0,
        "unexecuted_generated_calls": int(runs["unexecuted_generated_calls"].sum()) if len(runs) else 0,
        "unexpected_event_calls": int(runs["unexpected_event_calls"].sum()) if len(runs) else 0,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "tarball_sha256": sha256_file(tar_path),
    }
    if errors or len(runs) != len(manifest):
        raise RuntimeError(json.dumps(integrity, indent=2, ensure_ascii=False))
    return runs, decisions, overshoots, integrity


def design_slice(df: pd.DataFrame, design: str | None, level: str) -> pd.DataFrame:
    if design is None or level == "default":
        mask = pd.Series(True, index=df.index)
        for variable in DESIGNS:
            mask &= df[variable].eq(DEFAULTS[variable])
        return df[mask].copy()
    mask = df[design].eq(level)
    for other in DESIGNS:
        if other != design:
            mask &= df[other].eq(DEFAULTS[other])
    return df[mask].copy()


def apply_filters(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    out = df
    for key, value in filters.items():
        out = out[out[key].eq(value)]
    return out


def claim_effects(df: pd.DataFrame, claim: Claim, design: str | None = None, level: str = "default") -> pd.DataFrame:
    data = apply_filters(design_slice(df, design, level), claim.filters)
    contexts = [signal for signal in SIGNALS if signal != claim.arm and signal not in claim.filters]
    common = {"claim_id": claim.claim_id, "official_parent": claim.parent, "design_variable": design or "default_grid", "design_level": level}
    if claim.claim_type == "one_arm":
        rows = []
        for _, row in data.dropna(subset=[claim.metric]).iterrows():
            rows.append({
                **common,
                "seed": int(row["seed"]),
                "stratum_id": "|".join(f"{key}={row[key]}" for key in contexts),
                "signed_effect": float(row[claim.metric] - claim.threshold),
                "condition_id": row["condition_id"],
                "tar_probe_member": row["tar_probe_member"],
                **{key: row[key] for key in contexts},
            })
        return pd.DataFrame(rows)
    treatment = data[data[claim.arm].eq(claim.treatment)].dropna(subset=[claim.metric])
    reference = data[data[claim.arm].eq(claim.reference)].dropna(subset=[claim.metric])
    if treatment.empty or reference.empty:
        return pd.DataFrame()
    keys = ["seed"] + contexts
    t = treatment[keys + [claim.metric, "condition_id", "tar_probe_member"]].rename(columns={claim.metric: "treatment_value", "condition_id": "treatment_condition", "tar_probe_member": "treatment_tar_member"})
    r = reference[keys + [claim.metric, "condition_id", "tar_probe_member"]].rename(columns={claim.metric: "reference_value", "condition_id": "reference_condition", "tar_probe_member": "reference_tar_member"})
    merged = t.merge(r, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["signed_effect"] = merged["treatment_value"] - merged["reference_value"]
    merged["stratum_id"] = merged.apply(lambda row: "|".join(f"{key}={row[key]}" for key in contexts), axis=1)
    for key, value in common.items():
        merged[key] = value
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
        summary = summarize_effects(default, replicates, rng)
        design_points = []
        for design, levels in DESIGN_LEVELS.items():
            for level in levels:
                effects = claim_effects(df, claim, design, level)
                if effects.empty:
                    design_rows.append({"claim_id": claim.claim_id, "design_variable": design, "design_level": level, "identifiable": False})
                    continue
                effect_rows.append(effects)
                level_summary = summarize_effects(effects, replicates, rng)
                design_points.append(float(level_summary["point_estimate"]))
                design_rows.append({"claim_id": claim.claim_id, "design_variable": design, "design_level": level, "identifiable": True, **level_summary})
        if not design_points:
            cross = "not_identifiable"
        elif all(point > 0 for point in design_points):
            cross = "positive_direction_all_identifiable_cells"
        elif all(point <= 0 for point in design_points):
            cross = "nonpositive_all_identifiable_cells"
        else:
            cross = "mixed_across_identifiable_cells"
        summaries.append({
            "claim_id": claim.claim_id,
            "official_parent": claim.parent,
            "short_label": claim.label,
            **summary,
            "target_verdict": target_verdict(float(summary["ci_low"]), float(summary["ci_high"])),
            "cross_design_status": cross,
            "identifiable_nondefault_levels": len(design_points),
        })
    return pd.DataFrame(summaries), pd.DataFrame(design_rows), pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame()


def exposure_direction_by_factor(df: pd.DataFrame, factor: str, replicates: int, seed: int) -> pd.DataFrame:
    claim = next(claim for claim in CLAIMS if claim.claim_id == "Z1")
    effects = claim_effects(df, claim)
    rng = np.random.default_rng(seed)
    rows = []
    if factor not in effects.columns:
        return pd.DataFrame()
    for level, group in effects.groupby(factor, sort=True):
        summary = summarize_effects(group, replicates, rng)
        rows.append({"factor": factor, "level": level, **summary, "target_verdict": target_verdict(float(summary["ci_low"]), float(summary["ci_high"]))})
    return pd.DataFrame(rows)


def write_manifest(outdir: Path, inputs: list[Path], metadata: dict) -> None:
    files = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({"path": path.relative_to(outdir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (outdir / "ARTIFACT_MANIFEST.json").write_text(json.dumps({
        "generated_by": "analysis/polarization_cira_analysis.py",
        "inputs": [{"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs],
        "metadata": metadata,
        "files": files,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/polarization"))
    parser.add_argument("--tarball", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/polarization_sweep.tar.gz"))
    parser.add_argument("--assets-dir", type=Path, default=Path("references/socsim26_sharedtask/studies/polarization/scenario/assets"))
    parser.add_argument("--readme", type=Path, default=Path("references/socsim26_sharedtask/studies/polarization/README.md"))
    parser.add_argument("--sim-config", type=Path, default=Path("references/socsim26_sharedtask/studies/polarization/scenario/sim.yaml"))
    parser.add_argument("--runtime-action-prompts", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/runtime/prompts/action_prompts.py"))
    parser.add_argument("--runtime-resolver", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/environments/gm/components/resolve.py"))
    parser.add_argument("--runtime-turn-policy", type=Path, default=Path("references/silisocs_0_2_0/extracted/silisocs/simulation_engines/policies/turns.py"))
    parser.add_argument("--runtime-wheel", type=Path, default=Path("references/silisocs_0_2_0/silisocs-0.2.0-py3-none-any.whl"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts_polarization_final"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    source_audit = source_cap_audit(args.readme, args.sim_config, args.runtime_action_prompts, args.runtime_resolver, args.runtime_turn_policy, args.runtime_wheel)
    runs, decisions, overshoots, integrity = load_runs(args.study_dir, args.tarball, args.assets_dir)
    runs.to_csv(args.outdir / "run_level.csv", index=False)
    decisions.to_csv(args.outdir / "decision_level.csv", index=False)
    overshoots.to_csv(args.outdir / "action_cap_overshoots.csv", index=False)
    (args.outdir / "data_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    source_audit.update({
        "generated_overshoot_turns": integrity["generated_overshoot_turns"],
        "generated_calls_above_cap": integrity["generated_calls_above_cap"],
        "executed_overshoot_turns": integrity["executed_overshoot_turns"],
        "executed_events_above_cap": integrity["executed_events_above_cap"],
    })
    (args.outdir / "action_cap_source_audit.json").write_text(json.dumps(source_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    claims, design, effects = audit_claims(runs, args.bootstrap_replicates, args.seed)
    claims.to_csv(args.outdir / "hypothesis_audit.csv", index=False)
    design.to_csv(args.outdir / "design_identifiability.csv", index=False)
    effects.to_csv(args.outdir / "hypothesis_effect_rows.csv", index=False)
    model_direction = exposure_direction_by_factor(runs, "model", args.bootstrap_replicates, args.seed + 1)
    topic_direction = exposure_direction_by_factor(runs, "topic", args.bootstrap_replicates, args.seed + 2)
    model_direction.to_csv(args.outdir / "exposure_effect_by_model.csv", index=False)
    topic_direction.to_csv(args.outdir / "exposure_effect_by_topic.csv", index=False)
    by_model = runs.groupby("model", as_index=False).agg(
        runs=("condition_id", "size"),
        tool_call_decisions=("tool_call_decisions", "sum"),
        generated_overshoot_turns=("generated_overshoot_turns", "sum"),
        generated_calls_above_cap=("generated_calls_above_cap", "sum"),
        executed_overshoot_turns=("executed_overshoot_turns", "sum"),
        executed_events_above_cap=("executed_events_above_cap", "sum"),
        unexecuted_generated_calls=("unexecuted_generated_calls", "sum"),
    )
    by_model.to_csv(args.outdir / "execution_fidelity_by_model.csv", index=False)
    summary = {
        "study": "polarization",
        "runs": int(len(runs)),
        "conditions": int(runs["condition_id"].nunique()),
        "probe_calls": int(runs["probe_calls"].sum()),
        "tool_call_decisions": int(len(decisions)),
        "generated_overshoot_turns": integrity["generated_overshoot_turns"],
        "generated_calls_above_cap": integrity["generated_calls_above_cap"],
        "executed_overshoot_turns": integrity["executed_overshoot_turns"],
        "executed_events_above_cap": integrity["executed_events_above_cap"],
        "maximum_generated_calls_in_agent_episode": int(runs["maximum_generated_calls_in_agent_episode"].max()),
        "maximum_executed_events_in_agent_episode": int(runs["maximum_executed_events_in_agent_episode"].max()),
        "strict_probe_fidelity_nonmissing": integrity["strict_probe_fidelity_nonmissing"],
        "missing_probe_outputs": integrity["missing_probe_outputs"],
        "incomplete_probe_runs": integrity["incomplete_probe_runs"],
        "unexecuted_generated_calls": integrity["unexecuted_generated_calls"],
        "unexpected_event_calls": integrity["unexpected_event_calls"],
        "exact_action_sequence_runs": integrity["exact_action_sequence_runs"],
                "claims": claims.to_dict(orient="records"),
        "source_action_cap_audit": source_audit,
        "interpretation": "Nonmissing probe ratings are exact structured outputs, but two small-model runs contain failed probes. Multi-call batches can overshoot the advertised three-call cap, and some generated actions are rejected before becoming events; both are reported without imputing counterfactual trajectories.",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(args.outdir, [args.tarball, args.study_dir / "sweeps" / "manifest.csv", args.readme, args.sim_config, args.runtime_action_prompts, args.runtime_resolver, args.runtime_turn_policy, args.runtime_wheel], {"bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed, "runs": len(runs), "executed_overshoot_turns": integrity["executed_overshoot_turns"]})
    print(json.dumps({
        "runs": len(runs),
        "probe_calls": int(runs["probe_calls"].sum()),
        "tool_call_decisions": len(decisions),
        "generated_overshoot_turns": integrity["generated_overshoot_turns"],
        "generated_calls_above_cap": integrity["generated_calls_above_cap"],
        "executed_overshoot_turns": integrity["executed_overshoot_turns"],
        "executed_events_above_cap": integrity["executed_events_above_cap"],
        "claims": claims[["claim_id", "point_estimate", "ci_low", "ci_high", "target_verdict", "cross_design_status"]].to_dict(orient="records"),
        "model_direction": model_direction.to_dict(orient="records"),
        "topic_direction": topic_direction.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
