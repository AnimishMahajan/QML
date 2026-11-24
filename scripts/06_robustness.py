"""
Script 06: Evaluates model robustness against 5 perturbation families (Section 9, Section 12).
Applies perturbations to query images only (gallery clean).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from qdml.data.dataset import DefectCropDataset
from qdml.models.backbone import QDMLNet
from qdml.eval.pipeline import compute_retrieval_metrics
from qdml.quantum.distances import pairwise_quantum_similarity_matrix


PERTURBATION_LEVELS = {
    "brightness": [0.0, -0.2, -0.1, 0.1, 0.2, 0.3],  # additive offset
    "contrast": [1.0, 0.8, 0.6, 0.4, 1.3, 1.6],     # contrast factor
    "blur": [0.0, 0.5, 1.0, 2.0, 3.0],              # Gaussian radius
    "resolution": [1.0, 0.75, 0.5, 0.35, 0.25],      # downscale ratio
    "scale": [1.0, 0.8, 0.65, 1.25, 1.5],            # crop/pad scale
}


def apply_perturbation(img: Image.Image, family: str, severity: float) -> Image.Image:
    if family == "brightness":
        # factor = 1.0 + severity
        factor = max(0.1, 1.0 + severity)
        return ImageEnhance.Brightness(img).enhance(factor)
    elif family == "contrast":
        return ImageEnhance.Contrast(img).enhance(severity)
    elif family == "blur":
        if severity > 0:
            return img.filter(ImageFilter.GaussianBlur(severity))
        return img
    elif family == "resolution":
        if severity < 1.0:
            w, h = img.size
            dw, dh = max(8, int(w * severity)), max(8, int(h * severity))
            return img.resize((dw, dh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
        return img
    elif family == "scale":
        if severity != 1.0:
            w, h = img.size
            nw, nh = max(8, int(w * severity)), max(8, int(h * severity))
            resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
            # Center crop or pad back to (w, h)
            res = Image.new("RGB", (w, h), (0, 0, 0))
            x_off = max(0, (w - nw) // 2)
            y_off = max(0, (h - nh) // 2)
            crop_x = max(0, (nw - w) // 2)
            crop_y = max(0, (nh - h) // 2)
            cropped = resized.crop((crop_x, crop_y, crop_x + min(w, nw), crop_y + min(h, nh)))
            res.paste(cropped, (x_off, y_off))
            return res
        return img
    return img


def evaluate_model_robustness(
    model: QDMLNet,
    manifest_path: Path,
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    test_ds = DefectCropDataset(manifest_path, split="test", is_training=False)
    train_ds = DefectCropDataset(manifest_path, split="train", is_training=False)

    # 1. Extract clean gallery representations
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()

    gallery_feats = []
    gallery_states = []
    gallery_labels = []

    with torch.no_grad():
        for batch in train_loader:
            out = model(batch["image"].to(device))
            gallery_feats.append(out["x"].cpu().numpy())
            if out["psi"] is not None:
                gallery_states.append(out["psi"])
            gallery_labels.append(batch["label"].numpy())

    gal_x = np.concatenate(gallery_feats, axis=0)
    gal_labels = np.concatenate(gallery_labels, axis=0)
    gal_psi = torch.cat(gallery_states, dim=0) if gallery_states else None

    # Load raw PIL images for query
    query_raw_imgs = [test_ds._load_image(row["path"]) for _, row in test_ds.df.iterrows()]
    query_labels = test_ds.df["label"].values

    results = {}
    for family, severities in PERTURBATION_LEVELS.items():
        results[family] = {}
        for sev in severities:
            # Perturb queries
            perturbed_tensors = []
            for raw_img in query_raw_imgs:
                p_img = apply_perturbation(raw_img, family, sev)
                t = test_ds._to_tensor(p_img)
                perturbed_tensors.append(t)

            p_batch = torch.stack(perturbed_tensors).to(device)
            # Batched forward
            q_feats = []
            q_states = []
            with torch.no_grad():
                for idx in range(0, len(p_batch), 32):
                    sub_out = model(p_batch[idx:idx + 32])
                    q_feats.append(sub_out["x"].cpu().numpy())
                    if sub_out["psi"] is not None:
                        q_states.append(sub_out["psi"])

            qry_x = np.concatenate(q_feats, axis=0)
            qry_psi = torch.cat(q_states, dim=0) if q_states else None

            # Compute retrieval metrics
            if qry_psi is not None and gal_psi is not None:
                sims = pairwise_quantum_similarity_matrix("fidelity", qry_psi, gal_psi).cpu().numpy()
            else:
                sims = np.dot(qry_x, gal_x.T)

            metrics = compute_retrieval_metrics(sims, query_labels, gal_labels)
            results[family][str(sev)] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Robustness Profiles")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run directories")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="results/tables")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.data) / "manifest.csv"
    run_paths = [Path(p.strip()) for p in args.runs.split(",")]

    all_robustness = {}
    for rpath in run_paths:
        name = rpath.name
        cfg_path = rpath / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

        model = QDMLNet(
            embed_dim=cfg.get("embed_dim", 32),
            n_qubits=cfg.get("n_qubits", 6),
            n_layers=cfg.get("n_layers", 2),
            pretrained=False,
            slice_mode=cfg.get("slice_mode", "learned"),
            use_classical_head=cfg.get("use_classical_head", False),
        ).to(device)

        ckpt_path = rpath / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = rpath / "last.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        print(f"Running robustness evaluation for {name} on {device}...")
        rob = evaluate_model_robustness(model, manifest_path, device)
        all_robustness[name] = rob

    with open(out_dir / "table6_robustness.json", "w") as f:
        json.dump(all_robustness, f, indent=2)

    print(f"Robustness results saved to {out_dir / 'table6_robustness.json'}")


if __name__ == "__main__":
    main()
