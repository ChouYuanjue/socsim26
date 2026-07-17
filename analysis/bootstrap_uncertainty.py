from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics_module(path: Path):
    spec = importlib.util.spec_from_file_location("beauty_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/run_level.csv"))
    parser.add_argument("--metrics-module", type=Path, default=Path("analysis/beauty_metrics.py"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    module = load_metrics_module(args.metrics_module)
    run_df = pd.read_csv(args.runs)
    groups = [(condition_id, group.reset_index(drop=True)) for condition_id, group in run_df.groupby("condition_id", sort=True)]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for replicate in range(args.replicates):
        sampled_groups = []
        for _, group in groups:
            indices = rng.integers(0, len(group), size=len(group))
            sampled_groups.append(group.iloc[indices].copy())
        sampled = pd.concat(sampled_groups, ignore_index=True)
        summary = module.condition_summary(sampled)
        _, ratios = module.effect_audit(summary)
        for record in ratios.to_dict(orient="records"):
            rows.append({"replicate": replicate, **record})

    boot = pd.DataFrame(rows)
    point = pd.read_csv("artifacts/signal_variation_ratio.csv")
    ci_rows = []
    for metric, group in boot.groupby("metric", sort=False):
        values = group["signal_to_variation_ratio"].dropna()
        point_row = point[point["metric"] == metric].iloc[0]
        ci_rows.append({
            "metric": metric,
            "point_estimate": float(point_row["signal_to_variation_ratio"]),
            "bootstrap_mean": float(values.mean()),
            "ci_2_5": float(values.quantile(0.025)),
            "ci_50": float(values.quantile(0.5)),
            "ci_97_5": float(values.quantile(0.975)),
            "probability_svr_gt_1": float((values > 1).mean()),
            "replicates": int(len(values)),
        })
    ci = pd.DataFrame(ci_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    boot.to_csv(args.outdir / "svr_bootstrap_replicates.csv", index=False)
    ci.to_csv(args.outdir / "svr_bootstrap_ci.csv", index=False)
    (args.outdir / "summary.json").write_text(
        json.dumps({"seed": args.seed, "replicates": args.replicates, "metrics": ci_rows}, indent=2),
        encoding="utf-8",
    )
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
