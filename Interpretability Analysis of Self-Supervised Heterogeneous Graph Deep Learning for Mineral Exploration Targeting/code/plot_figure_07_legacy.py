"""Reproduce Figure 7 from the retained legacy local-IG table.

The legacy analysis contains 556 target-union anchors with complete
per-anchor Integrated Gradients and ten selected features.  This script
plots the stored long-form points; it does not require spatial input layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../plot_data/figure_07"))
    parser.add_argument("--output", type=Path, default=Path("../figures/figure_07/figure_7_legacy_local_ig.tif"))
    args = parser.parse_args()

    points = pd.read_csv(args.data_dir / "plot_points.csv")
    selected = pd.read_csv(args.data_dir / "feature_selection.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        subset = points[points["feature"] == row["feature"]]
        background = subset[~subset["known_deposit"]]
        deposits = subset[subset["known_deposit"]]
        ax.scatter(background["standardized_input"], background["integrated_gradient"],
                   c="#8a969c", s=28, alpha=0.35, linewidths=0)
        ax.scatter(deposits["standardized_input"], deposits["integrated_gradient"],
                   c="#e77d56", s=95, alpha=0.95, edgecolors="white", linewidths=1.0)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(f"({row['panel']}) {row['feature']}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Standardized input")
        ax.set_ylabel("Integrated Gradient")
        ax.grid(alpha=0.25)

    axes[0, 0].scatter([], [], c="#8a969c", s=75, alpha=0.55, label="Target-union anchors (n=556)")
    axes[0, 0].scatter([], [], c="#e77d56", s=95, edgecolors="white", label="Known deposits (n=31)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("SSL-GGL: Integrated Gradient dependence", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
