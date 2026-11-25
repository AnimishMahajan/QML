"""
Script 08: Explainability (XAI) Attribution on Image Pairs (Section 9 & Section 18).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from qdml.data.dataset import DefectCropDataset
from qdml.models.backbone import QDMLNet
from qdml.eval.xai import compute_pair_occlusion_map


def main():
    parser = argparse.ArgumentParser(description="Generate Pairwise Occlusion XAI")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    parser.add_argument("--n", type=int, default=3, help="Number of query samples to explain")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(args.data) / "manifest.csv"

    cfg_path = run_dir / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    ckpt_path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt"
    model = QDMLNet.load_from_checkpoint(ckpt_path, device=device, config_dict=cfg)

    train_ds = DefectCropDataset(manifest_path, split="train", is_training=False)
    test_ds = DefectCropDataset(manifest_path, split="test", is_training=False)

    print(f"Generating XAI maps for {args.n} queries on {device}...")

    for i in range(min(args.n, len(test_ds))):
        q_item = test_ds[i]
        q_tensor = q_item["image"]
        q_label = q_item["label"]

        # Find matching gallery item of same class
        gal_match_idx = None
        for j in range(len(train_ds)):
            if train_ds[j]["label"] == q_label:
                gal_match_idx = j
                break

        if gal_match_idx is None:
            continue

        g_item = train_ds[gal_match_idx]
        g_tensor = g_item["image"]

        heatmap = compute_pair_occlusion_map(
            model=model,
            query_tensor=q_tensor,
            gallery_tensor=g_tensor,
            patch_size=16,
            stride=8,
            device=device,
        )

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
        # Denormalize for plotting
        q_img = (q_tensor.permute(1, 2, 0).cpu().numpy() * 0.229 + 0.485).clip(0, 1)
        g_img = (g_tensor.permute(1, 2, 0).cpu().numpy() * 0.229 + 0.485).clip(0, 1)

        axes[0].imshow(q_img)
        axes[0].set_title(f"Query (Class {q_label})", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(g_img)
        axes[1].set_title(f"Gallery Match (Class {g_item['label']})", fontsize=9)
        axes[1].axis("off")

        im = axes[2].imshow(q_img)
        axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[2].set_title("Attribution Heatmap", fontsize=9)
        axes[2].axis("off")

        out_path = out_dir / f"fig12b_xai_sample_{i}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
