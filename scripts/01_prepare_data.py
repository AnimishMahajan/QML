"""
Script 01: Prepares raw VOC dataset into instance-level defect crops with 7:2:1 leak-free split.
"""

import argparse
from pathlib import Path
from qdml.data.prepare import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Substation Defect Dataset")
    parser.add_argument("--raw", type=str, default="dataset/Substation Equipment Appearance Defect Dataset/bdz-xml-6")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--context", type=float, default=0.15)
    parser.add_argument("--min-box", type=int, default=24)
    parser.add_argument("--save-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(f"Preparing dataset from: {args.raw}")
    df = prepare_dataset(
        raw_root=args.raw,
        out_dir=args.out,
        context_ratio=args.context,
        min_box=args.min_box,
        save_size=args.save_size,
        seed=args.seed,
    )
    print(f"Successfully processed {len(df)} defect crops into {args.out}.")


if __name__ == "__main__":
    main()
