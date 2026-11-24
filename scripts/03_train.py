"""
Script 03: Training entrypoint for Classical Baselines, AQPM-QDML, and CQHN-QDML.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from qdml.data.dataset import DefectCropDataset, PKSampler
from qdml.models.backbone import QDMLNet
from qdml.losses.aqpm import AQPMLoss
from qdml.losses.cqhn import CQHNLoss
from qdml.losses.baselines import (
    BatchHardTripletLoss,
    ContrastiveLoss,
    SupConLoss,
    CosFaceLoss,
)
from qdml.train.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss(method: str, cfg: dict) -> torch.nn.Module:
    m = method.lower()
    if m == "aqpm":
        return AQPMLoss(
            n_classes=cfg.get("n_classes", 5),
            n_qubits=cfg.get("n_qubits", 6),
            m0=cfg.get("m0", 0.2),
            gamma=cfg.get("gamma", 0.3),
            mp=cfg.get("mp", 0.5),
            lambda_p=cfg.get("lambda_p", 0.5),
            lambda_sep=cfg.get("lambda_sep", 0.5),
            lambda_r=cfg.get("lambda_r", 0.01),
            margin_form=cfg.get("margin_form", "ambiguity"),
            normalize_reliability=cfg.get("normalize_reliability", True),
            use_reliability_weight=cfg.get("use_reliability_weight", True),
        )
    elif m == "cqhn":
        return CQHNLoss(
            m0=cfg.get("m0", 0.2),
            gamma=cfg.get("gamma", 0.3),
            tau_a=cfg.get("tau_a", 0.5),
            lambda_inv=cfg.get("lambda_inv", 1.0),
            inv_warmup_epochs=cfg.get("inv_warmup_epochs", 5),
            lambda_struct=cfg.get("lambda_struct", 0.1),
            lambda_reg=cfg.get("lambda_reg", 0.05),
            positive_mode=cfg.get("positive_mode", "mixed"),
        )
    elif m in ("triplet", "baseline_triplet", "b2"):
        return BatchHardTripletLoss(margin=cfg.get("margin", 0.2))
    elif m in ("contrastive", "b1"):
        return ContrastiveLoss(margin=cfg.get("margin", 0.5))
    elif m in ("supcon", "b3"):
        return SupConLoss(temperature=cfg.get("temperature", 0.07))
    elif m in ("cosface", "b4"):
        return CosFaceLoss(in_features=cfg.get("embed_dim", 32), n_classes=cfg.get("n_classes", 5))
    else:
        raise ValueError(f"Unknown training method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Train QDML or Baseline model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--method", type=str, default="aqpm", help="aqpm, cqhn, triplet, contrastive, supcon, cosface")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--run-dir", type=str, default="results/runs/latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-k", type=int, default=6)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--slice-mode", type=str, default="learned")
    parser.add_argument("--use-classical-head", action="store_true")

    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    # CLI overrides
    method = cfg.get("method", args.method)
    seed = cfg.get("seed", args.seed)
    epochs = cfg.get("epochs", args.epochs)
    data_dir = Path(cfg.get("data_dir", args.data))
    run_dir = Path(cfg.get("run_dir", args.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing run for method='{method}' on device='{device}' (seed={seed})")

    manifest_path = data_dir / "manifest.csv"
    dual_views = (method.lower() == "cqhn")

    # Datasets
    train_dataset = DefectCropDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=224,
        is_training=True,
        dual_views=dual_views,
    )
    val_dataset = DefectCropDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=224,
        is_training=False,
        dual_views=False,
    )

    # PK Sampler: P=5 classes, K=6 instances -> Batch 30
    p = cfg.get("p", 5)
    k = cfg.get("k", args.batch_k)
    sampler = PKSampler(train_dataset.df["label"].values, p=p, k=k, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model
    model = QDMLNet(
        embed_dim=cfg.get("embed_dim", 32),
        n_qubits=cfg.get("n_qubits", args.n_qubits),
        n_layers=cfg.get("n_layers", args.n_layers),
        pretrained=cfg.get("pretrained", True),
        slice_mode=cfg.get("slice_mode", args.slice_mode),
        entangle_every_layer=cfg.get("entangle_every_layer", True),
        use_classical_head=cfg.get("use_classical_head", args.use_classical_head),
    )

    loss_fn = build_loss(method, {**cfg, "n_qubits": args.n_qubits, "n_layers": args.n_layers})

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        device=device,
        epochs=epochs,
        lr_classical=float(cfg.get("lr_classical", 3e-4)),
        lr_quantum=float(cfg.get("lr_quantum", 3e-2)),
        warmup_epochs=int(cfg.get("warmup_epochs", 2)),
    )

    fit_results = trainer.fit()

    # Save full config
    full_config = {
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "n_qubits": cfg.get("n_qubits", args.n_qubits),
        "n_layers": cfg.get("n_layers", args.n_layers),
        "slice_mode": cfg.get("slice_mode", args.slice_mode),
        "use_classical_head": cfg.get("use_classical_head", args.use_classical_head),
        "fit_results": fit_results,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    print(f"Training completed. Checkpoint saved to {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
