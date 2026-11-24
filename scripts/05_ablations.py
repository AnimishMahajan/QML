"""
Script 05: Executes ablation and baseline experiment grids across multiple random seeds.
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd


BASELINE_CONFIGS = {
    "B1_contrastive": {"method": "contrastive", "use_classical_head": True},
    "B2_triplet": {"method": "triplet", "use_classical_head": True},
    "B3_supcon": {"method": "supcon", "use_classical_head": True},
    "B4_cosface": {"method": "cosface", "use_classical_head": True},
    "B8_vqc_head": {"method": "triplet", "use_classical_head": False, "n_qubits": 6, "n_layers": 2},
}

AQPM_ABLATIONS = {
    "A0_full": {"method": "aqpm", "margin_form": "ambiguity", "normalize_reliability": True, "use_reliability_weight": True, "slice_mode": "learned"},
    "A1_no_reliability": {"method": "aqpm", "use_reliability_weight": False},
    "A3_fixed_margin": {"method": "aqpm", "margin_form": "fixed"},
    "A3b_brief_margin": {"method": "aqpm", "margin_form": "brief"},
    "A4b_unnorm_reliability": {"method": "aqpm", "normalize_reliability": False},
    "A6_classical_control": {"method": "triplet", "use_classical_head": True},
    "A9_topq_slice": {"method": "aqpm", "slice_mode": "topq"},
}

CQHN_ABLATIONS = {
    "C0_full": {"method": "cqhn", "slice_mode": "learned"},
    "C6_classical_control": {"method": "triplet", "use_classical_head": True},
}


def run_experiment(name: str, overrides: dict, seed: int, data_dir: str, epochs: int = 40):
    run_dir = Path(f"results/runs/{name}_s{seed}")
    eval_file = run_dir / "eval" / "evaluation.json"

    if eval_file.exists():
        print(f"Skipping already completed run: {run_dir}")
        with open(eval_file, "r") as f:
            return json.load(f)

    cmd = [
        sys.executable,
        "scripts/03_train.py",
        "--data", data_dir,
        "--run-dir", str(run_dir),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--method", overrides.get("method", "aqpm"),
    ]
    if overrides.get("use_classical_head", False):
        cmd.append("--use-classical-head")
    if "n_qubits" in overrides:
        cmd.extend(["--n-qubits", str(overrides["n_qubits"])])
    if "n_layers" in overrides:
        cmd.extend(["--n-layers", str(overrides["n_layers"])])
    if "slice_mode" in overrides:
        cmd.extend(["--slice-mode", overrides["slice_mode"]])

    print(f"\n--- Running {name} (seed {seed}) ---")
    subprocess.run(cmd, check=True)

    # Evaluate
    eval_cmd = [
        sys.executable,
        "scripts/04_evaluate.py",
        "--run", str(run_dir),
        "--data", data_dir,
    ]
    subprocess.run(eval_cmd, check=True)

    with open(eval_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Grids")
    parser.add_argument("--grid", type=str, default="aqpm", choices=["baselines", "aqpm", "cqhn", "resources"])
    parser.add_argument("--seeds", type=str, default="0", help="Comma separated seeds, e.g. 0,1,2")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    grid_dir = Path(f"results/grids/{args.grid}")
    grid_dir.mkdir(parents=True, exist_ok=True)

    if args.grid == "baselines":
        configs = BASELINE_CONFIGS
    elif args.grid == "aqpm":
        configs = AQPM_ABLATIONS
    elif args.grid == "cqhn":
        configs = CQHN_ABLATIONS
    else:
        configs = AQPM_ABLATIONS

    if args.only:
        selected = [k.strip() for k in args.only.split(",")]
        configs = {k: v for k, v in configs.items() if k in selected}

    aggregated = {}
    for name, cfg in configs.items():
        seed_results = []
        for s in seeds:
            res = run_experiment(name, cfg, seed=s, data_dir=args.data, epochs=args.epochs)
            # extract headline cross_split mAP and Recall@1
            c_ret = res["classical"]["cross_split"]
            q_ret = res.get("quantum", {}).get("cross_split", c_ret)
            seed_results.append({
                "seed": s,
                "classical_mAP": c_ret["mAP"],
                "classical_R1": c_ret["Recall@1"],
                "quantum_mAP": q_ret["mAP"],
                "quantum_R1": q_ret["Recall@1"],
            })

        c_maps = [x["classical_mAP"] for x in seed_results]
        q_maps = [x["quantum_mAP"] for x in seed_results]
        aggregated[name] = {
            "classical_mAP_mean": float(np.mean(c_maps)),
            "classical_mAP_std": float(np.std(c_maps)),
            "quantum_mAP_mean": float(np.mean(q_maps)),
            "quantum_mAP_std": float(np.std(q_maps)),
            "raw_seeds": seed_results,
        }

    with open(grid_dir / "aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    df = pd.DataFrame([
        {
            "Arm": k,
            "Classical mAP": f"{v['classical_mAP_mean']:.4f} ± {v['classical_mAP_std']:.4f}",
            "Quantum mAP": f"{v['quantum_mAP_mean']:.4f} ± {v['quantum_mAP_std']:.4f}",
        }
        for k, v in aggregated.items()
    ])
    df.to_markdown(grid_dir / "aggregated.md", index=False)
    print("\n" + "=" * 50)
    print(f"Grid '{args.grid}' Aggregated Results:")
    print(df.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    main()
