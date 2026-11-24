"""
Tests for loss functions and Findings F3, F4, F8.
"""

import pytest
import torch
from qdml.losses.aqpm import ReliabilityHead, AQPMLoss
from qdml.losses.cqhn import CQHNLoss


def test_reliability_normalisation_blocks_the_collapse():
    """Finding F3 verification: un-normalized reliability collapses to r -> 0 under gradient descent."""
    # 1. Un-normalized: loss = sum(r * hinge). With positive hinge, dL/dr >= 0 -> r goes to 0
    rel_unnorm = ReliabilityHead(embed_dim=4)
    x = torch.randn(8, 4)
    optim = torch.optim.SGD(rel_unnorm.parameters(), lr=1.0)
    fixed_hinge = torch.tensor(1.0)

    for _ in range(50):
        optim.zero_grad()
        r_tilde, raw_r = rel_unnorm(x, normalize=False)
        loss = torch.mean(r_tilde * fixed_hinge)
        loss.backward()
        optim.step()

    _, final_raw_unnorm = rel_unnorm(x, normalize=False)
    # Un-normalized raw_r is pushed towards 0
    assert final_raw_unnorm.mean().item() < 0.05

    # 2. Normalized: r_tilde = r / mean(r) has mean 1.0, so scaling all r changes nothing!
    rel_norm = ReliabilityHead(embed_dim=4)
    optim_norm = torch.optim.SGD(rel_norm.parameters(), lr=1.0)
    for _ in range(50):
        optim_norm.zero_grad()
        r_tilde, raw_r = rel_norm(x, normalize=True)
        # Adding small regularization lambda_r (mean(r) - 0.5)^2
        loss = torch.mean(r_tilde * fixed_hinge) + 0.01 * torch.square(torch.mean(raw_r) - 0.5)
        loss.backward()
        optim_norm.step()

    r_tilde_final, final_raw_norm = rel_norm(x, normalize=True)
    # Batch mean of r_tilde remains 1.0!
    assert pytest.approx(r_tilde_final.mean().item(), abs=1e-5) == 1.0
    assert final_raw_norm.mean().item() > 0.2


def test_hard_negative_mining_never_selects_a_same_class_partner():
    """Finding F8 verification: diagonal and same-class instances are masked."""
    cqhn = CQHNLoss()
    # Batch of 4 samples: class 0, class 0, class 1, class 1
    labels = torch.tensor([0, 0, 1, 1])
    # Synthetic similarity matrix: sample 0 is artificially identical to sample 1
    S_c = torch.tensor([
        [1.0, 0.99, 0.10, 0.20],
        [0.99, 1.0, 0.30, 0.15],
        [0.10, 0.30, 1.0, 0.95],
        [0.20, 0.15, 0.95, 1.0],
    ])

    hard_neg_idx, valid_neg, hard_pos_idx = cqhn._mine_pairs(S_c, labels)

    # For anchor 0 (class 0), hard negative must NOT be index 0 or index 1!
    assert hard_neg_idx[0].item() in (2, 3)
    assert hard_neg_idx[1].item() in (2, 3)
    assert hard_neg_idx[2].item() in (0, 1)
    assert hard_neg_idx[3].item() in (0, 1)
    assert valid_neg.all().item() is True


def test_single_class_batch_is_excluded_not_mislabelled():
    """Finding F8 verification: when a row has no valid negative, it is excluded."""
    cqhn = CQHNLoss()
    # All 4 samples belong to class 0
    labels = torch.tensor([0, 0, 0, 0])
    S_c = torch.ones((4, 4))

    hard_neg_idx, valid_neg, _ = cqhn._mine_pairs(S_c, labels)
    # No row has any valid negative
    assert not valid_neg.any().item()
