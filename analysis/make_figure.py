from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/signal_variation_ratio.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/signal_vs_design.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    labels = {
        "mean_choice": "Mean choice",
        "mean_reasoning_depth": "Reasoning depth",
        "mass_17_19": "Mass 17–19",
        "js_to_human": "JS distance",
        "tv_to_human": "TV distance",
        "emd_to_human": "EMD",
        "alignment_score": "Alignment",
    }
    df["label"] = df["metric"].map(labels).fillna(df["metric"])

    y = list(range(len(df)))
    height = 0.36
    fig, ax = plt.subplots(figsize=(4.7, 2.55))
    ax.barh([v + height / 2 for v in y], df["max_signal_effect"], height=height, label="Max signal effect")
    ax.barh([v - height / 2 for v in y], df["max_design_effect"], height=height, label="Max design effect")
    ax.set_yticks(y, df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Marginal effect range")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
