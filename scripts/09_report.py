"""
Script 09: Compiles comprehensive Markdown and LaTeX report with binding claim discipline.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from qdml.eval.resources import advantage_check


def main():
    parser = argparse.ArgumentParser(description="Generate Final Experiment Report")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out", type=str, default="results/REPORT.md")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_path = Path(args.out)

    # 1. Check claim discipline
    # Compare AQPM vs best classical baseline
    baselines_json = results_dir / "grids" / "baselines" / "aggregated.json"
    aqpm_json = results_dir / "grids" / "aqpm" / "aggregated.json"

    q_map, q_std, q_seeds = 0.0, 0.0, 0
    c_map, c_std, c_seeds = 0.0, 0.0, 0

    if aqpm_json.exists():
        with open(aqpm_json, "r") as f:
            aq_data = json.load(f)
            if "A0_full" in aq_data:
                q_map = aq_data["A0_full"]["quantum_mAP_mean"]
                q_std = aq_data["A0_full"]["quantum_mAP_std"]
                q_seeds = len(aq_data["A0_full"]["raw_seeds"])

    if baselines_json.exists():
        with open(baselines_json, "r") as f:
            b_data = json.load(f)
            # Find best classical baseline
            for b_name in ["B2_triplet", "B3_supcon", "B4_cosface"]:
                if b_name in b_data:
                    m = b_data[b_name]["classical_mAP_mean"]
                    if m > c_map:
                        c_map = m
                        c_std = b_data[b_name]["classical_mAP_std"]
                        c_seeds = len(b_data[b_name]["raw_seeds"])

    adv = advantage_check(
        quantum_mean=q_map,
        quantum_std=q_std,
        quantum_seeds=max(1, q_seeds),
        quantum_dim=6,
        quantum_params=24,
        classical_mean=c_map,
        classical_std=c_std,
        classical_seeds=max(1, c_seeds),
        classical_dim=32,
        classical_params=24,
    )

    report = []
    report.append("# QDML for Substation Defect Recognition: Final Experimental Report\n")
    report.append("## 1. Claim Discipline Verdict\n")
    if adv["advantage_claim_licensed"]:
        report.append("> **Verdict:** Quantum Advantage Claim is **LICENSED** under controlled protocol.\n")
    else:
        report.append("> **Verdict:** Quantum Advantage Claim is **NOT LICENSED**.\n")
        report.append(f"> **Approved Framing:** *{adv['permitted_framing']}*\n")
        report.append("> **Blocking Reasons:**")
        for r in adv["blocking_reasons"]:
            report.append(f"> - {r}")
        report.append("\n")

    report.append("## 2. Experimental Summary Tables\n")

    # Table 1
    t1_path = results_dir / "tables" / "table1_dataset.md"
    if t1_path.exists():
        with open(t1_path, "r") as f:
            report.append(f.read())
            report.append("\n")

    # Table 2
    b_grid_md = results_dir / "grids" / "baselines" / "aggregated.md"
    if b_grid_md.exists():
        report.append("### Table 2: Main Retrieval Comparison Across Arms\n")
        with open(b_grid_md, "r") as f:
            report.append(f.read())
            report.append("\n")

    # Table 4
    aq_grid_md = results_dir / "grids" / "aqpm" / "aggregated.md"
    if aq_grid_md.exists():
        report.append("### Table 4: AQPM Ablation Study (Novelties & Corrections)\n")
        with open(aq_grid_md, "r") as f:
            report.append(f.read())
            report.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(report))

    print(f"Report compiled to {out_path}.")
    print(f"Advantage Verdict: {adv['permitted_framing']}")


if __name__ == "__main__":
    main()
