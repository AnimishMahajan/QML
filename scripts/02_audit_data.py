"""
Script 02: Audit dataset integrity, verify leak-free partitions, and emit Table 1.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Audit Processed Dataset")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="results/tables")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run 01_prepare_data.py first.")

    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} total defect crops.")

    # 1. Leakage check
    source_splits = df.groupby("source_image")["split"].nunique()
    leaks = int((source_splits > 1).sum())
    is_leak_free = (leaks == 0)
    print(f"Leakage Check: {leaks} source frames in multiple splits (Pass: {is_leak_free})")

    # 2. Split counts and class distributions
    table1 = pd.crosstab(df["class_name"], df["split"], margins=True)
    print("\nTable 1 — Class Distribution Across Splits:")
    print(table1.to_string())

    # Check all classes present in all splits
    classes_in_train = set(df[df["split"] == "train"]["class_name"].unique())
    classes_in_val = set(df[df["split"] == "val"]["class_name"].unique())
    classes_in_test = set(df[df["split"] == "test"]["class_name"].unique())
    all_classes = set(df["class_name"].unique())

    all_present = (classes_in_train == all_classes) and (classes_in_val == all_classes) and (classes_in_test == all_classes)
    print(f"All classes present in all three splits: {all_present}")

    # 3. Fraction of small defects (< 2% of frame)
    small_defects = (df["scale_frac"] < 0.02).mean() * 100.0
    print(f"Fraction of defects occupying < 2% of frame area: {small_defects:.2f}%")

    # 4. Save Table 1 artefacts
    table1.to_csv(out_dir / "table1_dataset.csv")
    with open(out_dir / "table1_dataset.md", "w") as f:
        f.write("# Table 1: Dataset and Class Distribution\n\n")
        f.write(table1.to_markdown())
        f.write(f"\n\n**Integrity Verdict:** Zero split leakage = {is_leak_free}. Defect area < 2%: {small_defects:.1f}%.")

    audit_summary = {
        "n_crops": len(df),
        "n_source_frames": int(df["source_image"].nunique()),
        "is_leak_free": is_leak_free,
        "all_classes_present_all_splits": all_present,
        "small_defect_percentage": small_defects,
        "split_ratio": (df["split"].value_counts(normalize=True)).to_dict(),
    }
    with open(out_dir / "dataset_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Audit artefacts written to {out_dir}.")


if __name__ == "__main__":
    main()
