"""
CQHN-QDML: Condition-aware Quantum Hard-Negative Deep Metric Learning Objective (Section 7).
"""

from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantum.statevector import state_fidelity, pairwise_state_fidelity


class CQHNLoss(nn.Module):
    """
    CQHN-QDML Loss Module incorporating:
      1. Condition invariance loss with warmup ramp (F7).
      2. Classically-guided hard-negative mining with diagonal/positive masking (F8).
      3. Mixed positive pair selection (F6).
      4. Ambiguity-adaptive margin.
      5. Similarity-structure consistency.
      6. Active read-out regularizer on raw expectations (F11).
    """

    def __init__(
        self,
        m0: float = 0.2,
        gamma: float = 0.3,
        tau_a: float = 0.5,
        lambda_inv: float = 1.0,
        inv_warmup_epochs: int = 5,
        lambda_struct: float = 0.1,
        lambda_reg: float = 0.05,
        positive_mode: Literal["mixed", "same_class", "self_view"] = "mixed",
    ):
        super().__init__()
        self.m0 = m0
        self.gamma = gamma
        self.tau_a = tau_a
        self.lambda_inv_target = lambda_inv
        self.inv_warmup_epochs = inv_warmup_epochs
        self.lambda_struct = lambda_struct
        self.lambda_reg = lambda_reg
        self.positive_mode = positive_mode

    def get_lambda_inv(self, epoch: int) -> float:
        """F7: Linear ramp of lambda_inv to prevent early embedding collapse."""
        if self.inv_warmup_epochs <= 0:
            return self.lambda_inv_target
        progress = min(1.0, float(epoch + 1) / float(self.inv_warmup_epochs))
        return progress * self.lambda_inv_target

    def _mine_pairs(
        self,
        S_c: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        F8: Mines hardest negatives using classical cosine similarity.
        Masks the diagonal and same-class instances. Returns valid mask so invalid
        rows are excluded rather than defaulting to index 0 (which trains on a positive).
        """
        B = labels.shape[0]
        device = labels.device

        # Same-class mask (B, B)
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)

        # 1. Hardest negative: argmax over j where y_j != y_i and j != i
        S_neg = S_c.clone()
        S_neg[label_matrix] = -float("inf")
        S_neg[diag_mask] = -float("inf")

        valid_negatives = (S_neg > -float("inf")).any(dim=1)
        hard_neg_indices = torch.argmax(S_neg, dim=1)

        # 2. Hardest positive partner (for mixed positive mode, F6)
        S_pos = S_c.clone()
        S_pos[~label_matrix] = -float("inf")
        S_pos[diag_mask] = -float("inf")

        has_pos_partner = (S_pos > -float("inf")).any(dim=1)
        hard_pos_indices = torch.argmax(S_pos, dim=1)

        return hard_neg_indices, valid_negatives, hard_pos_indices

    def forward(
        self,
        psi_a: torch.Tensor,
        psi_b: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        z_raw_a: torch.Tensor,
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Dual view inputs from symmetric condition distribution:
          psi_a, psi_b: (B, 2**q)
          x_a, x_b: (B, 32)
          z_raw_a: (B, out_dim)
          labels: (B,)
        """
        B = labels.shape[0]
        device = labels.device

        # 1. Condition Invariance Loss L_inv
        # L_inv = (1/N) sum_i [1 - F(psi_i^(a), psi_i^(b))]
        fid_views = state_fidelity(psi_a, psi_b)
        loss_inv = torch.mean(1.0 - fid_views)

        # 2. Classical similarity matrix S_c = x_a @ x_a^T
        S_c = torch.matmul(x_a, x_a.t())

        # 3. Classically-guided hard negative mining (F8)
        hard_neg_idx, valid_neg, hard_pos_idx = self._mine_pairs(S_c, labels)

        # 4. Determine positive pair distance D_q(i, i^+) (F6)
        # psi_anchor = psi_a
        D_pos_list = []
        for i in range(B):
            use_self = False
            if self.positive_mode == "self_view":
                use_self = True
            elif self.positive_mode == "same_class":
                use_self = False
            else:  # mixed (default)
                use_self = (i % 2 == 0)

            if use_self or hard_pos_idx[i] == i:
                # Compare anchor to self under condition view b
                f_pos = state_fidelity(psi_a[i:i + 1], psi_b[i:i + 1])
            else:
                partner_idx = hard_pos_idx[i]
                f_pos = state_fidelity(psi_a[i:i + 1], psi_a[partner_idx:partner_idx + 1])
            D_pos_list.append(1.0 - f_pos.squeeze())
        D_pos = torch.stack(D_pos_list)

        # 5. Quantum distance to hardest negative D_q(i, j_i)
        neg_psi = psi_a[hard_neg_idx]
        D_neg = 1.0 - state_fidelity(psi_a, neg_psi)

        # 6. Ambiguity-adaptive margin
        # a_i = S_c[i, j_i]
        a_i = S_c[torch.arange(B, device=device), hard_neg_idx]
        margin = self.m0 + self.gamma * F.relu(a_i - self.tau_a)

        # 7. Hard-negative hinge loss (exclude invalid rows without negatives, F8)
        hinge = F.relu(D_pos - D_neg + margin)
        if valid_neg.sum() > 0:
            loss_hn = torch.mean(hinge[valid_neg])
        else:
            loss_hn = torch.tensor(0.0, device=device)

        # 8. Local-to-global similarity consistency L_struct
        # S_q = pairwise fidelities on batch a
        S_q = pairwise_state_fidelity(psi_a)
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        if B > 1:
            s_c_off = S_c[mask]
            s_q_off = S_q[mask]
            norm_c = (s_c_off - s_c_off.mean()) / (s_c_off.std() + 1e-8)
            norm_q = (s_q_off - s_q_off.mean()) / (s_q_off.std() + 1e-8)
            loss_struct = torch.mean(torch.square(norm_c - norm_q))
        else:
            loss_struct = torch.tensor(0.0, device=device)

        # 9. Read-out regularizer on raw expectations (F11)
        z_norm = torch.norm(z_raw_a, p=2, dim=-1)
        loss_reg = torch.mean(torch.square(z_norm - 1.0))

        # Total scheduled loss
        curr_lambda_inv = self.get_lambda_inv(epoch)
        total_loss = (
            loss_hn
            + curr_lambda_inv * loss_inv
            + self.lambda_struct * loss_struct
            + self.lambda_reg * loss_reg
        )

        stats = {
            "loss_hn": loss_hn.item(),
            "loss_inv": loss_inv.item(),
            "loss_struct": loss_struct.item(),
            "loss_reg": loss_reg.item(),
            "total_loss": total_loss.item(),
            "lambda_inv": curr_lambda_inv,
            "mean_D_pos": D_pos.mean().item(),
            "mean_D_neg": D_neg.mean().item(),
            "valid_negatives": int(valid_neg.sum().item()),
            "mean_ambiguity": a_i.mean().item(),
        }
        return total_loss, stats
