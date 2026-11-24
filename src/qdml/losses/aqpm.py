"""
AQPM-QDML: Adaptive Quantum Prototype-Metric Deep Learning Objective (Section 6).
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantum.statevector import (
    density_matrix_from_states,
    state_to_density_fidelity,
    density_distance_trace,
    density_purity,
)


class ReliabilityHead(nn.Module):
    """
    Estimates sample reliability r_i = sigmoid(w_r^T x_i + b_r).
    F3 Fix: normalized by batch mean r_tilde = r / mean(r).
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_r = torch.sigmoid(self.fc(x)).squeeze(-1)  # (B,) in (0, 1)
        if normalize:
            # F3: normalize by batch mean so optimizer cannot simply drive r -> 0
            mean_r = torch.mean(raw_r) + 1e-8
            r_tilde = raw_r / mean_r
        else:
            r_tilde = raw_r
        return r_tilde, raw_r


class PrototypeBank(nn.Module):
    """
    Maintains density-matrix prototypes rho_c for C classes with EMA updating (F10).
    """

    def __init__(self, n_classes: int = 5, n_qubits: int = 6, ema_decay: float = 0.9):
        super().__init__()
        self.n_classes = n_classes
        self.dim = 1 << n_qubits
        self.ema_decay = ema_decay

        # Register persistent buffers for EMA density matrices: (C, D, D) complex
        initial_bank = torch.zeros((n_classes, self.dim, self.dim), dtype=torch.complex64)
        for c in range(n_classes):
            # initialize to maximally mixed state I / D
            initial_bank[c] = torch.eye(self.dim, dtype=torch.complex64) / float(self.dim)
        self.register_buffer("bank", initial_bank)
        self.register_buffer("initialized", torch.zeros(n_classes, dtype=torch.bool))

    def update(self, class_idx: int, current_rho: torch.Tensor):
        """EMA update for class prototype."""
        with torch.no_grad():
            if not self.initialized[class_idx]:
                self.bank[class_idx] = current_rho.detach()
                self.initialized[class_idx] = True
            else:
                updated = self.ema_decay * self.bank[class_idx] + (1.0 - self.ema_decay) * current_rho.detach()
                self.bank[class_idx] = updated


class AQPMLoss(nn.Module):
    """
    Complete AQPM-QDML loss module with prototype compactness and separation.
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_qubits: int = 6,
        m0: float = 0.2,
        gamma: float = 0.3,
        mp: float = 0.5,
        lambda_p: float = 0.5,
        lambda_sep: float = 0.5,
        lambda_r: float = 0.01,
        margin_form: Literal["ambiguity", "brief", "fixed"] = "ambiguity",
        normalize_reliability: bool = True,
        use_reliability_weight: bool = True,  # A1 ablation
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.m0 = m0
        self.gamma = gamma
        self.mp = mp
        self.lambda_p = lambda_p
        self.lambda_sep = lambda_sep
        self.lambda_r = lambda_r
        self.margin_form = margin_form
        self.normalize_reliability = normalize_reliability
        self.use_reliability_weight = use_reliability_weight

        self.rel_head = ReliabilityHead(embed_dim=32)
        self.proto_bank = PrototypeBank(n_classes=n_classes, n_qubits=n_qubits)

    def forward(
        self,
        psi: torch.Tensor,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        psi: (B, 2**q) complex quantum state vectors
        x: (B, 32) classical embeddings
        labels: (B,) class labels in [0, C-1]
        """
        B = psi.shape[0]
        device = psi.device

        # 1. Estimate reliability r_i
        r_tilde, raw_r = self.rel_head(x, normalize=self.normalize_reliability)
        if not self.use_reliability_weight:
            # Ablation A1: unweighted prototypes and loss
            r_weights = torch.ones_like(r_tilde)
        else:
            r_weights = r_tilde

        # 2. Build current-batch density-matrix prototypes rho_c (with gradient)
        batch_rhos = {}
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_states = psi[mask]
                c_weights = r_weights[mask]
                rho_c = density_matrix_from_states(c_states, c_weights)
                batch_rhos[c] = rho_c
                # Update EMA bank
                self.proto_bank.update(c, rho_c)
            else:
                # Class absent from batch: fall back to EMA bank
                batch_rhos[c] = self.proto_bank.bank[c]

        # 3. Compute sample-to-prototype distances D_ic = 1 - <psi_i | rho_c | psi_i>
        D_matrix = torch.zeros((B, self.n_classes), device=device)
        for c in range(self.n_classes):
            s_ic = state_to_density_fidelity(psi, batch_rhos[c])
            D_matrix[:, c] = 1.0 - s_ic

        # 4. Extract positive and hardest negative prototype distances
        D_pos = D_matrix[torch.arange(B, device=device), labels]

        # Mask same-class to find hardest negative
        D_neg_matrix = D_matrix.clone()
        D_neg_matrix[torch.arange(B, device=device), labels] = float("inf")
        D_neg, hard_neg_class = torch.min(D_neg_matrix, dim=-1)
        S_hard = 1.0 - D_neg

        # 5. Adaptive margin m_i (F4 / D1)
        if self.margin_form == "ambiguity":
            margin = self.m0 + self.gamma * S_hard
        elif self.margin_form == "brief":
            margin = self.m0 + self.gamma * (1.0 - S_hard)
        elif self.margin_form == "fixed":
            margin = torch.full_like(S_hard, self.m0)
        else:
            raise ValueError(f"Unknown margin_form: {self.margin_form}")

        # 6. Reliability-gated metric loss
        # L_AQPM = (1/N) sum_i r_tilde_i * [D_pos - D_neg + m_i]_+
        hinge = F.relu(D_pos - D_neg + margin)
        loss_aqpm = torch.mean(r_weights * hinge)

        # 7. Prototype compactness and separation
        loss_compact = torch.mean(r_weights * D_pos)

        # Separation between distinct prototype pairs: d(rho_c, rho_k) = 1 - Tr(rho_c rho_k)/...
        loss_sep = torch.tensor(0.0, device=device)
        pair_count = 0
        for c in range(self.n_classes):
            for k in range(c + 1, self.n_classes):
                d_ck = density_distance_trace(batch_rhos[c], batch_rhos[k])
                loss_sep = loss_sep + F.relu(self.mp - d_ck)
                pair_count += 1
        if pair_count > 0:
            loss_sep = loss_sep / float(pair_count)

        loss_proto = loss_compact + self.lambda_sep * loss_sep

        # 8. Regularization on raw reliability to prevent saturation
        loss_r = self.lambda_r * torch.square(torch.mean(raw_r) - 0.5)

        total_loss = loss_aqpm + self.lambda_p * loss_proto + loss_r

        # Metrics for logging
        purities = [density_purity(batch_rhos[c]).item() for c in range(self.n_classes)]
        active_frac = (hinge > 0).float().mean().item()

        stats = {
            "loss_aqpm": loss_aqpm.item(),
            "loss_proto": loss_proto.item(),
            "loss_compact": loss_compact.item(),
            "loss_sep": loss_sep.item(),
            "loss_r": loss_r.item(),
            "total_loss": total_loss.item(),
            "r_mean": raw_r.mean().item(),
            "r_std": raw_r.std().item() if B > 1 else 0.0,
            "active_frac": active_frac,
            "mean_D_pos": D_pos.mean().item(),
            "mean_D_neg": D_neg.mean().item(),
            "mean_purity": sum(purities) / len(purities),
        }
        return total_loss, stats
