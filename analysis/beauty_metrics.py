from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

ANCHORS = {
    "basic": {11: .04, 12: .00, 13: .03, 14: .06, 15: .01, 16: .06, 17: .32, 18: .30, 19: .12, 20: .06},
    "cycle": {11: .01, 12: .01, 13: .00, 14: .01, 15: .00, 16: .04, 17: .10, 18: .22, 19: .47, 20: .13},
    "costless": {11: .00, 12: .04, 13: .00, 14: .04, 15: .04, 16: .04, 17: .09, 18: .21, 19: .40, 20: .15},
}
CHOICES = list(range(11, 21))
VARIABLES = [
    "game_variant",
    "goal_framing",
    "model",
    "persona",
    "instruction_wording",
    "response_format",
    "temperature",
    "persona_format",
]
SIGNAL_VARIABLES = ["game_variant", "goal_framing", "model", "persona"]
DESIGN_VARIABLES = ["instruction_wording", "response_format", "temperature", "persona_format"]
DEFAULTS = {
    "instruction_wording": "default",
    "response_format": "default",
    "temperature": "0.5",
    "persona_format": "default",
}
METRICS = [
    "mean_choice",
    "mean_reasoning_depth",
    "mass_17_19",
    "js_to_human",
    "tv_to_human",
    "emd_to_human",
    "alignment_score",
]


def normalize(values: Iterable[float]) -> list[float]:
    values = list(values)
    total = sum(values)
    return [value / total for value in values] if total > 0 else [0.0 for _ in values]


def js_distance(p: list[float], q: list[float]) -> float:
    """Square-root Jensen-Shannon divergence, bounded in [0, 1] with log base 2."""
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


def load_toolkit(shared_repo: Path) -> None:
    sys.path.insert(0, str(shared_repo.resolve()))


def parse_condition_id(condition_id: str) -> dict[str, str]:
    labels = dict(DEFAULTS)
    for segment in str(condition_id).split("__"):
        if "-" not in segment:
            continue
        key, value = segment.split("-", 1)
        if key in VARIABLES:
            labels[key] = value
    missing = [key for key in SIGNAL_VARIABLES if key not in labels]
    if missing:
        raise ValueError(f"Condition id missing required variables {missing}: {condition_id}")
    return labels


def run_level_rows(study) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    failures: list[dict] = []
    for run in study.runs():
        labels = parse_condition_id(run.condition_id)
        try:
            values = [
                int(obs["value"])
                for obs in run.observables()
                if obs.get("obs_type") == "game_choice" and obs.get("value") is not None
            ]
        except Exception as exc:
            failures.append({
                "condition_id": run.condition_id,
                "seed": run.seed,
                "kind": run.kind,
                **labels,
                "reason": f"observable_error:{type(exc).__name__}:{exc}",
            })
            continue
        if not values:
            failures.append({
                "condition_id": run.condition_id,
                "seed": run.seed,
                "kind": run.kind,
                **labels,
                "reason": "no_game_choice_observable",
            })
            continue
        counts = Counter(values)
        row = {
            "condition_id": run.condition_id,
            "seed": run.seed,
            "kind": run.kind,
            **labels,
            "n_choices": len(values),
            "mean_choice": sum(values) / len(values),
            "mean_reasoning_depth": sum(20 - value for value in values) / len(values),
            "mass_17_19": sum(17 <= value <= 19 for value in values) / len(values),
        }
        for choice in CHOICES:
            row[f"count_{choice}"] = counts.get(choice, 0)
        rows.append(row)
    return rows, failures


def condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for condition_id, group in run_df.groupby("condition_id", sort=True):
        first = group.iloc[0]
        row = {
            "condition_id": condition_id,
            "kind": first["kind"],
            **{variable: first[variable] for variable in VARIABLES},
        }
        counts = [float(group[f"count_{choice}"].sum()) for choice in CHOICES]
        distribution = normalize(counts)
        row.update({f"p_{choice}": probability for choice, probability in zip(CHOICES, distribution)})
        row["runs"] = int(len(group))
        row["choices"] = int(sum(counts))
        row["mean_choice"] = sum(choice * probability for choice, probability in zip(CHOICES, distribution))
        row["mean_reasoning_depth"] = 20 - row["mean_choice"]
        row["mass_17_19"] = sum(distribution[choice - 11] for choice in (17, 18, 19))
        variant = row["game_variant"]
        anchor = [ANCHORS[variant][choice] for choice in CHOICES]
        row["js_to_human"] = js_distance(distribution, anchor)
        row["tv_to_human"] = total_variation(distribution, anchor)
        row["emd_to_human"] = earth_mover_1d(distribution, anchor)
        row["alignment_score"] = 1 - (
            row["js_to_human"] + row["tv_to_human"] + row["emd_to_human"]
        ) / 3
        rows.append(row)
    return pd.DataFrame(rows)


def signal_effect_rows(summary_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    grid = summary_df[summary_df["kind"] == "grid"]
    for variable in SIGNAL_VARIABLES:
        for metric in METRICS:
            grouped = grid.groupby(variable, dropna=False)[metric].mean().dropna()
            if len(grouped) < 2:
                continue
            rows.append({
                "variable": variable,
                "role": "signal",
                "metric": metric,
                "levels": int(len(grouped)),
                "min_group_mean": float(grouped.min()),
                "max_group_mean": float(grouped.max()),
                "effect_range": float(grouped.max() - grouped.min()),
                "group_means_json": json.dumps({str(k): float(v) for k, v in grouped.items()}),
            })
    return rows


def matched_design_subset(summary_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    changed = summary_df[(summary_df["kind"] == "variation") & (summary_df[variable] != DEFAULTS[variable])]
    if changed.empty:
        return changed
    signal_keys = changed[SIGNAL_VARIABLES].drop_duplicates()
    baseline = summary_df[(summary_df["kind"] == "grid") & (summary_df[variable] == DEFAULTS[variable])]
    baseline = baseline.merge(signal_keys, on=SIGNAL_VARIABLES, how="inner")
    baseline_columns = list(summary_df.columns)
    baseline = baseline[baseline_columns]
    return pd.concat([baseline, changed], ignore_index=True)


def design_effect_rows(summary_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for variable in DESIGN_VARIABLES:
        matched = matched_design_subset(summary_df, variable)
        if matched.empty:
            continue
        for metric in METRICS:
            grouped = matched.groupby(variable, dropna=False)[metric].mean().dropna()
            if len(grouped) < 2:
                continue
            rows.append({
                "variable": variable,
                "role": "design_variation",
                "metric": metric,
                "levels": int(len(grouped)),
                "min_group_mean": float(grouped.min()),
                "max_group_mean": float(grouped.max()),
                "effect_range": float(grouped.max() - grouped.min()),
                "group_means_json": json.dumps({str(k): float(v) for k, v in grouped.items()}),
            })
    return rows


def effect_audit(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.DataFrame(signal_effect_rows(summary_df) + design_effect_rows(summary_df))
    ratios: list[dict] = []
    for metric in METRICS:
        signal = audit[(audit["role"] == "signal") & (audit["metric"] == metric)]["effect_range"]
        design = audit[(audit["role"] == "design_variation") & (audit["metric"] == metric)]["effect_range"]
        max_signal = float(signal.max()) if len(signal) else math.nan
        max_design = float(design.max()) if len(design) else math.nan
        ratios.append({
            "metric": metric,
            "max_signal_effect": max_signal,
            "max_design_effect": max_design,
            "signal_to_variation_ratio": max_signal / max_design if max_design > 0 else math.nan,
        })
    return audit, pd.DataFrame(ratios)


def planned_contrasts(summary_df: pd.DataFrame) -> pd.DataFrame:
    grid = summary_df[summary_df["kind"] == "grid"]
    specs = {
        "game_variant": [("cycle", "basic"), ("costless", "basic")],
        "goal_framing": [("strategic", "none")],
        "persona": [("intuitive", "neutral"), ("cautious", "neutral"), ("competitive", "neutral")],
        "model": [
            ("qwen3.5-9b", "qwen3.5-4b"),
            ("qwen3.5-27b-fp8", "qwen3.5-4b"),
            ("gemma-4-31b", "qwen3.5-27b-fp8"),
        ],
    }
    rows: list[dict] = []
    for variable, pairs in specs.items():
        grouped = grid.groupby(variable)[METRICS].mean()
        for treatment, reference in pairs:
            if treatment not in grouped.index or reference not in grouped.index:
                continue
            for metric in METRICS:
                rows.append({
                    "variable": variable,
                    "treatment": treatment,
                    "reference": reference,
                    "metric": metric,
                    "treatment_mean": float(grouped.loc[treatment, metric]),
                    "reference_mean": float(grouped.loc[reference, metric]),
                    "difference": float(grouped.loc[treatment, metric] - grouped.loc[reference, metric]),
                })
    return pd.DataFrame(rows)


def main_variant_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df[
        (summary_df["kind"] == "grid")
        & (summary_df["goal_framing"] == "none")
        & (summary_df["persona"] == "neutral")
    ]
    return baseline.groupby("game_variant", as_index=False).agg(
        conditions=("condition_id", "size"),
        choices=("choices", "sum"),
        mean_choice=("mean_choice", "mean"),
        mean_reasoning_depth=("mean_reasoning_depth", "mean"),
        mass_17_19=("mass_17_19", "mean"),
        js_to_human=("js_to_human", "mean"),
        tv_to_human=("tv_to_human", "mean"),
        emd_to_human=("emd_to_human", "mean"),
        alignment_score=("alignment_score", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributional signal-variation audit for beauty_contest")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument("--shared-repo", type=Path, default=Path("references/socsim26_sharedtask"))
    args = parser.parse_args()

    load_toolkit(args.shared_repo)
    from socsim_eval import open_study

    study = open_study(str(args.study))
    rows, failures = run_level_rows(study)
    if not rows:
        raise RuntimeError("No game_choice observables found; verify extraction and study path")

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)
    run_df.to_csv(args.outdir / "run_level.csv", index=False)
    failure_df.to_csv(args.outdir / "unparsed_runs.csv", index=False)

    summary_df = condition_summary(run_df)
    summary_df.to_csv(args.outdir / "condition_summary.csv", index=False)

    audit_df, ratio_df = effect_audit(summary_df)
    audit_df.to_csv(args.outdir / "effect_audit.csv", index=False)
    ratio_df.to_csv(args.outdir / "signal_variation_ratio.csv", index=False)
    planned_contrasts(summary_df).to_csv(args.outdir / "planned_contrasts.csv", index=False)
    main_variant_table(summary_df).to_csv(args.outdir / "main_variant_table.csv", index=False)

    expected_runs = len(list(study.runs()))
    overview = {
        "expected_runs": expected_runs,
        "parsed_runs": int(len(run_df)),
        "unparsed_runs": int(len(failure_df)),
        "parse_rate": len(run_df) / expected_runs if expected_runs else None,
        "conditions": int(len(summary_df)),
        "mean_choice": float(run_df["mean_choice"].mean()),
        "mean_reasoning_depth": float(run_df["mean_reasoning_depth"].mean()),
        "outputs": [str(path) for path in sorted(args.outdir.glob("*.csv"))],
    }
    (args.outdir / "summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
