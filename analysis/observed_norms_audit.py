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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from final_cira_analysis import sha256_file, summarize_effects, target_verdict

QUESTIONS = ["q1_family_importance", "q182_homosexuality", "q188_euthanasia"]
RANGES = {"q1_family_importance": (1, 4), "q182_homosexuality": (1, 10), "q188_euthanasia": (1, 10)}


def read_jsonl(tar: tarfile.TarFile, member: str) -> list[dict]:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    raw = handle.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as z:
        text = z.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_released_integer_parser(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "parse_integer_answer"]
    if len(selected) != 1:
        raise RuntimeError("parse_integer_answer not found")
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    node = selected[0]
    lines = source.splitlines()
    audit = {
        "source_path": source_path.as_posix(),
        "sha256": sha256_file(source_path),
        "function": "parse_integer_answer",
        "start_line": int(node.lineno),
        "end_line": int(node.end_lineno or node.lineno),
        "source": "\n".join(lines[node.lineno - 1 : int(node.end_lineno or node.lineno)]),
        "execution_mode": "exact AST definition compiled with standard-library re",
    }
    return namespace["parse_integer_answer"], audit


def strict_integer(output: str, lo: int, hi: int) -> int | None:
    text = str(output or "").strip()
    if not re.fullmatch(r"\d+", text):
        return None
    value = int(text)
    return value if lo <= value <= hi else None


def labels_from_condition(condition_id: str, model: str) -> dict[str, str]:
    labels = {
        "population": "",
        "condition": "",
        "model": model,
        "framing": "neutral",
        "scale_labels": "default",
    }
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in labels:
            labels[key] = value
    if not labels["population"] or not labels["condition"]:
        raise ValueError(f"cannot parse {condition_id}")
    return labels


def load_runs(study_dir: Path, tar_path: Path, released_parser) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = study_dir / "sweeps" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"seed": int})
    probe_rows = []
    run_rows = []
    errors = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = set(tar.getnames())
        for raw in manifest.itertuples(index=False):
            row = raw._asdict()
            condition_id = str(row["condition_id"])
            labels = labels_from_condition(condition_id, str(row["model"]))
            run_dir = str(row["run_dir"]).replace("\\", "/").strip("/")
            prefix = f"observed_norms/{run_dir}/run"
            event_member = f"{prefix}/probe_events.jsonl.gz"
            prompt_member = f"{prefix}/prompts_and_responses.jsonl.gz"
            try:
                if event_member not in members or prompt_member not in members:
                    raise FileNotFoundError(f"missing {event_member} or {prompt_member}")
                events = read_jsonl(tar, event_member)
                prompts = [p for p in read_jsonl(tar, prompt_member) if p.get("phase") == "probe"]
                if len(events) != len(prompts):
                    raise ValueError(f"event/prompt length mismatch {len(events)}/{len(prompts)}")
                values: dict[tuple[str, str], list[int]] = defaultdict(list)
                for index, (event, prompt) in enumerate(zip(events, prompts)):
                    question = str(event["label"])
                    lo, hi = RANGES[question]
                    output = str(prompt.get("output", ""))
                    replay = released_parser(output, lo=lo, hi=hi)
                    strict = strict_integer(output, lo, hi)
                    recorded = int(event["data"]["probe_return"])
                    if replay != recorded:
                        raise ValueError(f"released parser replay mismatch at probe {index}")
                    if str(event.get("source_user")) != str(prompt.get("agent_name")):
                        raise ValueError(f"agent order mismatch at probe {index}")
                    turn = str(event["data"]["turn"])
                    values[(question, turn)].append(recorded)
                    probe_rows.append({
                        "condition_id": condition_id,
                        "kind": str(row["kind"]),
                        "seed": int(row["seed"]),
                        "probe_index": index,
                        "agent_name": str(event.get("source_user")),
                        "question": question,
                        "turn": turn,
                        **labels,
                        "recorded_value": recorded,
                        "released_parser_value": replay,
                        "strict_integer_value": strict,
                        "strict_fidelity": int(strict == recorded),
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "raw_output": output,
                        "tar_event_member": event_member,
                        "tar_prompt_member": prompt_member,
                    })
                compression_by_question = {}
                mean_shift_by_question = {}
                if labels["condition"] == "observed":
                    for question in QUESTIONS:
                        t0 = np.asarray(values[(question, "t0")], dtype=float)
                        t1 = np.asarray(values[(question, "t1")], dtype=float)
                        compression_by_question[question] = float(np.var(t0, ddof=1) - np.var(t1, ddof=1))
                        mean_shift_by_question[question] = float(np.mean(t1) - np.mean(t0))
                    compression = float(np.mean(list(compression_by_question.values())))
                    absolute_mean_shift = float(np.mean([abs(value) for value in mean_shift_by_question.values()]))
                else:
                    compression = math.nan
                    absolute_mean_shift = math.nan
                run_rows.append({
                    "condition_id": condition_id,
                    "kind": str(row["kind"]),
                    "seed": int(row["seed"]),
                    "model": str(row["model"]),
                    **labels,
                    "probe_calls": len(events),
                    "dispersion_compression": compression,
                    "absolute_mean_shift": absolute_mean_shift,
                    **{f"compression_{question}": compression_by_question.get(question, math.nan) for question in QUESTIONS},
                    **{f"mean_shift_{question}": mean_shift_by_question.get(question, math.nan) for question in QUESTIONS},
                    "tar_event_member": event_member,
                    "tar_prompt_member": prompt_member,
                })
            except Exception as exc:
                errors.append({"condition_id": condition_id, "seed": int(row["seed"]), "type": type(exc).__name__, "error": str(exc)})
    runs = pd.DataFrame(run_rows)
    probes = pd.DataFrame(probe_rows)
    integrity = {
        "manifest_rows": int(len(manifest)),
        "parsed_runs": int(len(runs)),
        "probe_calls": int(len(probes)),
        "released_parser_replay_accuracy": float((probes["released_parser_value"] == probes["recorded_value"]).mean()) if len(probes) else math.nan,
        "strict_integer_fidelity": float(probes["strict_fidelity"].mean()) if len(probes) else math.nan,
        "exact_integer_outputs": int(probes["strict_fidelity"].sum()) if len(probes) else 0,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "tarball_sha256": sha256_file(tar_path),
    }
    if errors or len(runs) != len(manifest):
        raise RuntimeError(json.dumps(integrity, indent=2, ensure_ascii=False))
    return runs, probes, integrity


def compression_effects(runs: pd.DataFrame) -> pd.DataFrame:
    data = runs[runs["condition"] == "observed"].dropna(subset=["dispersion_compression"]).copy()
    rows = []
    context_columns = ["population", "model", "framing", "scale_labels"]
    for _, row in data.iterrows():
        rows.append({
            "seed": int(row["seed"]),
            "stratum_id": "|".join(f"{key}={row[key]}" for key in context_columns),
            "signed_effect": float(row["dispersion_compression"]),
            **{key: row[key] for key in context_columns},
            "condition_id": row["condition_id"],
            "tar_event_member": row["tar_event_member"],
        })
    return pd.DataFrame(rows)


def country_rank_stability(runs: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    observed = probes[(probes["condition"] == "observed") & (probes["framing"] == "neutral") & (probes["scale_labels"] == "default")]
    rows = []
    for (model, question, seed), group in observed.groupby(["model", "question", "seed"]):
        means = group.groupby(["turn", "population"])["recorded_value"].mean().unstack("population")
        if set(means.index) >= {"t0", "t1"}:
            t0 = means.loc["t0"]
            t1 = means.loc["t1"]
            constant_profile = bool(t0.nunique(dropna=True) <= 1 or t1.nunique(dropna=True) <= 1)
            correlation = math.nan if constant_profile else float(t0.corr(t1, method="spearman"))
            rows.append({
                "model": model,
                "question": question,
                "seed": int(seed),
                "spearman_t0_t1": correlation,
                "undefined_due_to_constant_profile": constant_profile,
            })
    return pd.DataFrame(rows)


def write_manifest(outdir: Path, inputs: list[Path], metadata: dict) -> None:
    files = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append({"path": path.relative_to(outdir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (outdir / "ARTIFACT_MANIFEST.json").write_text(json.dumps({
        "generated_by": "analysis/observed_norms_audit.py",
        "inputs": [{"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs],
        "metadata": metadata,
        "files": files,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/observed_norms"))
    parser.add_argument("--tarball", type=Path, default=Path("references/socsim26_sharedtask/socsim26_data/observed_norms_sweep.tar.gz"))
    parser.add_argument("--parser-source", type=Path, default=Path("references/socsim26_sharedtask/task_components/norms.py"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts_observed_norms_final"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    released_parser, parser_audit = load_released_integer_parser(args.parser_source)
    runs, probes, integrity = load_runs(args.study_dir, args.tarball, released_parser)
    runs.to_csv(args.outdir / "run_level.csv", index=False)
    probes.to_csv(args.outdir / "probe_level.csv", index=False)
    (args.outdir / "data_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    parser_audit.update({
        "probe_calls": int(len(probes)),
        "released_parser_replay_accuracy": integrity["released_parser_replay_accuracy"],
        "strict_integer_fidelity": integrity["strict_integer_fidelity"],
    })
    (args.outdir / "parser_source_audit.json").write_text(json.dumps(parser_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    effects = compression_effects(runs)
    rng = np.random.default_rng(args.seed)
    compression_summary = summarize_effects(effects, args.bootstrap_replicates, rng)
    compression_result = {
        "claim_id": "N3",
        "official_parent": "h3_interaction_compression",
        "short_label": "observation reduces within-population response dispersion",
        **compression_summary,
        "target_verdict": target_verdict(float(compression_summary["ci_low"]), float(compression_summary["ci_high"])),
    }
    pd.DataFrame([compression_result]).to_csv(args.outdir / "identifiable_hypothesis_audit.csv", index=False)
    effects.to_csv(args.outdir / "dispersion_compression_effect_rows.csv", index=False)
    rank = country_rank_stability(runs, probes)
    rank.to_csv(args.outdir / "country_rank_stability.csv", index=False)
    rank_summary = rank.groupby(["model", "question"], as_index=False).agg(mean_spearman=("spearman_t0_t1", "mean"), minimum_spearman=("spearman_t0_t1", "min"))
    rank_summary.to_csv(args.outdir / "country_rank_stability_summary.csv", index=False)

    nonidentifiable = pd.DataFrame([
        {"official_parent": "h1_country_direction", "status": "not_identifiable_from_released_package", "reason": "registered WVS human anchors are not distributed because of licensing"},
        {"official_parent": "h2_private_homogenization", "status": "not_identifiable_from_released_package", "reason": "claim compares model dispersion with WVS respondent dispersion, which is not distributed"},
        {"official_parent": "h5_model_capability", "status": "not_identifiable_from_released_package", "reason": "directional accuracy is defined relative to unavailable WVS anchors"},
    ])
    nonidentifiable.to_csv(args.outdir / "nonidentifiable_official_claims.csv", index=False)
    summary = {
        "study": "observed_norms",
        "runs": int(len(runs)),
        "probe_calls": int(len(probes)),
        "strict_integer_fidelity": integrity["strict_integer_fidelity"],
        "released_parser_replay_accuracy": integrity["released_parser_replay_accuracy"],
        "identifiable_claim": compression_result,
        "nonidentifiable_official_claims": nonidentifiable.to_dict(orient="records"),
        "interpretation": "Observed Norms is a negative execution-control: every output is exactly one in-range integer. Human-fidelity claims remain unavailable without licensed WVS anchors; only within-simulation observation effects are directly auditable.",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(args.outdir, [args.tarball, args.study_dir / "sweeps" / "manifest.csv", args.parser_source], {
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "runs": len(runs),
        "probe_calls": len(probes),
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
