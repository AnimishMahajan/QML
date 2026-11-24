"""
Trainer implementation for QDML with separate optimizer groups, validation checkpointing,
and quantum health monitoring.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.backbone import QDMLNet
from ..losses.aqpm import AQPMLoss
from ..losses.cqhn import CQHNLoss
from ..eval.pipeline import compute_retrieval_metrics


class Trainer:
    """
    Standard Trainer supporting Classical Baselines, AQPM-QDML, and CQHN-QDML.
    """

    def __init__(
        self,
        model: QDMLNet,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: str | Path,
        device: torch.device,
        epochs: int = 40,
        lr_classical: float = 3e-4,
        lr_quantum: float = 3e-2,
        warmup_epochs: int = 2,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs

        # 1. Separate optimizer parameter groups (F9)
        optim_groups = self.model.get_optim_groups(lr_backbone=lr_classical, lr_quantum=lr_quantum)
        # Add loss_fn parameters (e.g. reliability head or CosFace weights)
        loss_params = list(self.loss_fn.parameters())
        if len(loss_params) > 0:
            optim_groups.append({"params": loss_params, "lr": lr_classical})

        self.optimizer = torch.optim.Adam(optim_groups)

        # 2. Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        self.warmup_epochs = warmup_epochs

        self.history: List[Dict[str, float]] = []
        self.best_val_map = -1.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        # Handle warmup
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / float(self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * factor

        epoch_stats: Dict[str, List[float]] = {}

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            labels = batch["label"].to(self.device)

            if isinstance(self.loss_fn, CQHNLoss):
                img_a = batch["image_a"].to(self.device)
                img_b = batch["image_b"].to(self.device)

                out_a = self.model(img_a)
                out_b = self.model(img_b)

                loss, stats = self.loss_fn(
                    psi_a=out_a["psi"],
                    psi_b=out_b["psi"],
                    x_a=out_a["x"],
                    x_b=out_b["x"],
                    z_raw_a=out_a["z_raw"],
                    labels=labels,
                    epoch=epoch,
                )
            elif isinstance(self.loss_fn, AQPMLoss):
                images = batch["image"].to(self.device)
                out = self.model(images)
                loss, stats = self.loss_fn(
                    psi=out["psi"],
                    x=out["x"],
                    labels=labels,
                )
            else:
                # Classical baseline
                images = batch["image"].to(self.device)
                out = self.model(images)
                loss, stats = self.loss_fn(out["x"], labels)

            loss.backward()
            self.optimizer.step()

            for k, v in stats.items():
                if k not in epoch_stats:
                    epoch_stats[k] = []
                epoch_stats[k].append(v)

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        # Summary of training epoch
        summary = {f"train_{k}": float(np.mean(v)) for k, v in epoch_stats.items()}
        return summary

    def evaluate_validation(self) -> Dict[str, float]:
        """Evaluates retrieval performance on the validation split."""
        self.model.eval()

        feats = []
        lbls = []
        srcs = []

        with torch.no_grad():
            for batch in self.val_loader:
                img = batch["image"].to(self.device)
                out = self.model(img)
                # If quantum model, evaluate on classical embedding and/or state vector
                x = out["x"].cpu().numpy()
                feats.append(x)
                lbls.append(batch["label"].numpy())
                srcs.append(np.array(batch["source_image"]))

        all_feats = np.concatenate(feats, axis=0)
        all_lbls = np.concatenate(lbls, axis=0)
        all_srcs = np.concatenate(srcs, axis=0)

        # Leave-one-out validation retrieval
        sim_matrix = np.dot(all_feats, all_feats.T)
        np.fill_diagonal(sim_matrix, -np.inf)

        metrics = compute_retrieval_metrics(
            similarity_matrix=sim_matrix,
            query_labels=all_lbls,
            gallery_labels=all_lbls,
            query_sources=all_srcs,
            gallery_sources=all_srcs,
            exclude_same_source=True,
        )
        return {f"val_{k}": v for k, v in metrics.items()}

    def fit(self) -> Dict[str, float]:
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(self.epochs):
            t_train = self.train_epoch(epoch)
            t_val = self.evaluate_validation()

            log_entry = {"epoch": epoch + 1, **t_train, **t_val}
            self.history.append(log_entry)

            val_map = t_val.get("val_mAP", 0.0)
            if val_map > self.best_val_map:
                self.best_val_map = val_map
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch + 1, "best_mAP": val_map},
                    self.run_dir / "best.pt",
                )

            # Save last checkpoint
            torch.save(
                {"model": self.model.state_dict(), "epoch": epoch + 1, "last_mAP": val_map},
                self.run_dir / "last.pt",
            )

            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1:2d}/{self.epochs} - Loss: {t_train.get('train_total_loss', 0.0):.4f} - Val mAP: {val_map:.4f} (Best: {self.best_val_map:.4f})")

        total_time = time.time() - start_time
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return {"best_val_mAP": self.best_val_map, "training_time_sec": total_time}
