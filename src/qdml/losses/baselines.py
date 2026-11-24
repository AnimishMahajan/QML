"""
Classical Metric Learning Baseline Losses (B1 - B6).
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """B1: Pairwise contrastive loss."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # embeddings: (B, d) normalized
        dist = torch.cdist(embeddings, embeddings, p=2)
        B = labels.shape[0]
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        diag = torch.eye(B, dtype=torch.bool, device=embeddings.device)

        pos_dist = dist[label_matrix.bool() & ~diag]
        neg_dist = dist[(~label_matrix.bool()) & ~diag]

        loss_pos = torch.mean(torch.square(pos_dist)) if len(pos_dist) > 0 else torch.tensor(0.0)
        loss_neg = torch.mean(torch.square(F.relu(self.margin - neg_dist))) if len(neg_dist) > 0 else torch.tensor(0.0)
        loss = 0.5 * (loss_pos + loss_neg)
        return loss, {"loss_contrastive": loss.item()}


class BatchHardTripletLoss(nn.Module):
    """B2: Batch-hard triplet loss."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Euclidean distance on L2-normalized embeddings
        dist = torch.cdist(embeddings, embeddings, p=2)
        B = labels.shape[0]
        device = embeddings.device

        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0))
        diff_class = ~same_class
        diag = torch.eye(B, dtype=torch.bool, device=device)

        # Hardest positive: maximum distance to same class (excluding self)
        dist_pos = dist.clone()
        dist_pos[~same_class | diag] = -float("inf")
        hard_pos, _ = torch.max(dist_pos, dim=1)

        # Hardest negative: minimum distance to different class
        dist_neg = dist.clone()
        dist_neg[~diff_class] = float("inf")
        hard_neg, _ = torch.min(dist_neg, dim=1)

        valid = (hard_pos > -float("inf")) & (hard_neg < float("inf"))
        hinge = F.relu(hard_pos - hard_neg + self.margin)
        loss = torch.mean(hinge[valid]) if valid.sum() > 0 else torch.tensor(0.0, device=device)
        return loss, {"loss_triplet": loss.item(), "mean_pos_dist": hard_pos[valid].mean().item()}


class SupConLoss(nn.Module):
    """B3: Supervised Contrastive Loss (Khosla et al.)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = embeddings.shape[0]
        device = embeddings.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Cosine similarity matrix
        sim = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(B).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss, {"loss_supcon": loss.item()}


class CosFaceLoss(nn.Module):
    """B4: Large Margin Cosine Loss (CosFace)."""

    def __init__(self, in_features: int = 32, n_classes: int = 5, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Normalize weights and embeddings
        norm_w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, norm_w)
        phi = cosine - self.m

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = F.cross_entropy(output, labels)
        return loss, {"loss_cosface": loss.item()}
