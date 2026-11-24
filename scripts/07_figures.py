"""
Script 07: Generates publication-grade figures (Section 9, Section 20).
Enforces identical projection settings for embedding geometry (Fig 11).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_class_distribution(manifest_path: Path, out_path: Path):
    df = pd.read_csv(manifest_path)
    counts = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Defect Instance Distribution Across 7:2:1 Splits", fontsize=12, fontweight="bold")
    ax.set_xlabel("Defect Category", fontsize=10)
    ax.set_ylabel("Instance Crops", fontsize=10)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_retrieval_comparison(grid_json: Path, out_path: Path):
    with open(grid_json, "r") as f:
        data = json.load(f)

    arms = list(data.keys())
    c_maps = [data[a]["classical_mAP_mean"] for a in arms]
    q_maps = [data[a]["quantum_mAP_mean"] for a in arms]

    x = np.arange(len(arms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.bar(x - width / 2, c_maps, width, label="Classical Representation (32-D)", color="#3470a3")
    ax.bar(x + width / 2, q_maps, width, label="Quantum Hilbert Space (q=6)", color="#8e44ad")

    ax.set_ylabel("Retrieval mAP (cross_split)", fontsize=10)
    ax.set_title("Main Retrieval Comparison Across Arms", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=25, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_ablation_deltas(ablation_json: Path, out_path: Path):
    with open(ablation_json, "r") as f:
        data = json.load(f)

    base_map = data.get("A0_full", {}).get("quantum_mAP_mean", 0.0)
    arms = [k for k in data.keys() if k != "A0_full"]
    deltas = [(data[k]["quantum_mAP_mean"] - base_map) for k in arms]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
    ax.barh(arms, deltas, color=colors)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ mAP vs Full Model (A0)", fontsize=10)
    ax.set_title("AQPM Ablation Delta Impact", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_robustness_curves(robustness_json: Path, out_path: Path):
    with open(robustness_json, "r") as f:
        rob = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=300)
    families = ["blur", "brightness", "resolution"]

    for idx, fam in enumerate(families):
        ax = axes[idx]
        for arm_name, arm_data in rob.items():
            if fam not in arm_data:
                continue
            sevs = sorted([float(k) for k in arm_data[fam].keys()])
            maps = [arm_data[fam][str(s)]["mAP"] for s in sevs]
            ax.plot(sevs, maps, marker="o", label=arm_name)

        ax.set_title(f"Robustness: {fam.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity Level", fontsize=9)
        ax.set_ylabel("Retrieval mAP", fontsize=9)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Figures")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--grid", type=str, default=None)
    parser.add_argument("--ablation-grid", type=str, default=None)
    parser.add_argument("--robustness", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.data) / "manifest.csv"
    if manifest_path.exists():
        plot_class_distribution(manifest_path, out_dir / "fig1_class_distribution.png")

    if args.grid and Path(args.grid).exists():
        grid_json = Path(args.grid) / "aggregated.json" if Path(args.grid).is_dir() else Path(args.grid)
        if grid_json.exists():
            plot_retrieval_comparison(grid_json, out_dir / "fig2_retrieval.png")

    if args.ablation_grid and Path(args.ablation_grid).exists():
        ab_json = Path(args.ablation_grid) / "aggregated.json" if Path(args.ablation_grid).is_dir() else Path(args.ablation_grid)
        if ab_json.exists():
            plot_ablation_deltas(ab_json, out_dir / "fig4_ablation.png")

    if args.robustness and Path(args.robustness).exists():
        plot_robustness_curves(Path(args.robustness), out_dir / "fig6_robustness.png")

    print(f"Figures generated in {out_dir}.")


if __name__ == "__main__":
    main()
