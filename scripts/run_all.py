"""
Master Runner: Executes end-to-end pipeline covering:
  - Step 0: Environment & CUDA Verification
  - Step 1: Data Preparation & Audit
  - Step 2: Classical Metric Learning Baselines (B1 Contrastive, B2 Triplet, B3 SupCon, B4 CosFace)
  - Step 3: Quantum Similarity Baseline (B8 VQC Head)
  - Step 4: Proposed AQPM-QDML (Training, Evaluation, Ablation Grid)
  - Step 5: Proposed CQHN-QDML (Training, Evaluation, Ablation Grid)
  - Step 6: Condition-Shift Robustness Evaluation
  - Step 7: Figures, XAI Attribution Maps & Final Advantage Verification Report

Supports --smoke for rapid 2-epoch validation across all components.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list, desc: str):
    print("\n" + "=" * 70)
    print(f"STEP: {desc}")
    print(f"COMMAND: {' '.join(str(c) for c in cmd)}")
    print("=" * 70)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="QDML Complete End-to-End Pipeline Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch smoke test across all stages")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds (e.g. '0' or '0,1,2')")
    parser.add_argument("--data", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation and audit if already done")
    args = parser.parse_args()

    epochs = 2 if args.smoke else args.epochs
    seeds_list = [s.strip() for s in args.seeds.split(",")]
    primary_seed = seeds_list[0]
    data_dir = args.data
    manifest_path = Path(data_dir) / "manifest.csv"

    # =========================================================================
    # Step 0: Environment & CUDA Check
    # =========================================================================
    run_step([sys.executable, "scripts/00_check_env.py"], "Check Environment & CUDA Acceleration")

    # =========================================================================
    # Step 1: Data Preparation & Audit
    # =========================================================================
    if not args.skip_prep and not manifest_path.exists():
        run_step([sys.executable, "scripts/01_prepare_data.py"], "Prepare Dataset Crops (VOC -> 224x224 crops)")
        run_step([sys.executable, "scripts/02_audit_data.py"], "Audit Dataset & Verify Integrity")
    else:
        print(f"\n[INFO] Dataset manifest found at '{manifest_path}'. Skipping data prep.")

    # =========================================================================
    # Step 2: Baseline Triplet Model Evaluation & Training
    # =========================================================================
    triplet_run_dir = Path(f"results/runs/triplet_s{primary_seed}")
    triplet_ckpt = triplet_run_dir / "best.pt"
    if not triplet_ckpt.exists():
        run_step([
            sys.executable, "scripts/03_train.py",
            "--config", "configs/baseline_triplet.yaml",
            "--epochs", str(epochs),
            "--seed", primary_seed,
            "--data", data_dir,
            "--run-dir", str(triplet_run_dir),
        ], f"Train Classical Triplet Baseline (seed {primary_seed})")

    run_step([
        sys.executable, "scripts/04_evaluate.py",
        "--run", str(triplet_run_dir),
        "--data", data_dir,
    ], f"Evaluate Classical Triplet Baseline (Retrieval & Classification P/R/F1)")

    # =========================================================================
    # Step 3: Classical Metric Learning Baselines (B1, B3, B4)
    # =========================================================================
    run_step([
        sys.executable, "scripts/05_ablations.py",
        "--grid", "baselines",
        "--seeds", args.seeds,
        "--only", "B1_contrastive,B3_supcon,B4_cosface",
        "--data", data_dir,
        "--epochs", str(epochs),
    ], f"Train & Evaluate Classical Baselines (B1 Contrastive, B3 SupCon, B4 CosFace)")

    # =========================================================================
    # Step 4: Quantum Similarity Baseline (B8 VQC Head)
    # =========================================================================
    run_step([
        sys.executable, "scripts/05_ablations.py",
        "--grid", "baselines",
        "--seeds", args.seeds,
        "--only", "B8_vqc_head",
        "--data", data_dir,
        "--epochs", str(epochs),
    ], "Train & Evaluate Quantum Similarity Baseline (B8 VQC Head)")

    # =========================================================================
    # Step 5: Proposed Method 1 — AQPM-QDML
    # =========================================================================
    aqpm_run_dir = Path(f"results/runs/aqpm_s{primary_seed}")
    aqpm_ckpt = aqpm_run_dir / "best.pt"
    if not aqpm_ckpt.exists():
        run_step([
            sys.executable, "scripts/03_train.py",
            "--config", "configs/aqpm.yaml",
            "--epochs", str(epochs),
            "--seed", primary_seed,
            "--data", data_dir,
            "--run-dir", str(aqpm_run_dir),
        ], f"Train Proposed AQPM-QDML (seed {primary_seed})")

    run_step([
        sys.executable, "scripts/04_evaluate.py",
        "--run", str(aqpm_run_dir),
        "--data", data_dir,
    ], f"Evaluate AQPM-QDML on Test Split (Fidelity Retrieval & Classification)")

    # Run AQPM ablation grid
    aqpm_only = "A0_full,A1_no_reliability,A6_classical_control,A9_topq_slice" if args.smoke else None
    aqpm_ablation_cmd = [
        sys.executable, "scripts/05_ablations.py",
        "--grid", "aqpm",
        "--seeds", args.seeds,
        "--data", data_dir,
        "--epochs", str(epochs),
    ]
    if aqpm_only:
        aqpm_ablation_cmd.extend(["--only", aqpm_only])

    run_step(aqpm_ablation_cmd, "Run AQPM Ablation Grid")

    # =========================================================================
    # Step 6: Proposed Method 2 — CQHN-QDML
    # =========================================================================
    cqhn_run_dir = Path(f"results/runs/cqhn_s{primary_seed}")
    cqhn_ckpt = cqhn_run_dir / "best.pt"
    if not cqhn_ckpt.exists():
        run_step([
            sys.executable, "scripts/03_train.py",
            "--config", "configs/cqhn.yaml",
            "--epochs", str(epochs),
            "--seed", primary_seed,
            "--data", data_dir,
            "--run-dir", str(cqhn_run_dir),
        ], f"Train Proposed CQHN-QDML (seed {primary_seed})")

    run_step([
        sys.executable, "scripts/04_evaluate.py",
        "--run", str(cqhn_run_dir),
        "--data", data_dir,
    ], f"Evaluate CQHN-QDML on Test Split")

    # Run CQHN ablation grid
    run_step([
        sys.executable, "scripts/05_ablations.py",
        "--grid", "cqhn",
        "--seeds", args.seeds,
        "--data", data_dir,
        "--epochs", str(epochs),
    ], "Run CQHN Ablation Grid")

    # =========================================================================
    # Step 7: Condition-Shift Robustness Evaluation
    # =========================================================================
    candidate_runs = [triplet_run_dir, aqpm_run_dir, cqhn_run_dir]
    available_runs = [str(r) for r in candidate_runs if (r / "best.pt").exists() or (r / "last.pt").exists()]
    if available_runs:
        run_step([
            sys.executable, "scripts/06_robustness.py",
            "--runs", ",".join(available_runs),
            "--data", data_dir,
        ], f"Evaluate Condition-Shift Robustness on: {available_runs}")

    # =========================================================================
    # Step 8: Visualizations, Explainability & Final Verification Report
    # =========================================================================
    # 8a: Figures
    fig_cmd = [
        sys.executable, "scripts/07_figures.py",
        "--data", data_dir,
        "--grid", "results/grids/baselines",
        "--ablation-grid", "results/grids/aqpm",
    ]
    rob_table = Path("results/tables/table6_robustness.json")
    if rob_table.exists():
        fig_cmd.extend(["--robustness", str(rob_table)])
    run_step(fig_cmd, "Generate Publication Figures (Distribution, Retrieval, Deltas)")

    # 8b: Explainability (XAI)
    xai_target = aqpm_run_dir if aqpm_ckpt.exists() else triplet_run_dir
    run_step([
        sys.executable, "scripts/08_xai.py",
        "--run", str(xai_target),
        "--data", data_dir,
    ], f"Generate Pairwise Occlusion XAI Heatmaps using {xai_target.name}")

    # 8c: Final Report & Claim Discipline Verification
    run_step([
        sys.executable, "scripts/09_report.py",
        "--results", "results",
        "--out", "results/REPORT.md",
    ], "Compile Final Report & Run Advantage Claim Discipline Check")

    print("\n" + "*" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("Final report generated at: results/REPORT.md")
    print("*" * 70)


if __name__ == "__main__":
    main()
