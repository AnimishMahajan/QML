"""
Integration tests: End-to-end forward/backward step on QDMLNet with AQPM and CQHN.
"""

import pytest
import torch
from qdml.models.backbone import QDMLNet
from qdml.losses.aqpm import AQPMLoss
from qdml.losses.cqhn import CQHNLoss


def test_aqpm_forward_backward():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QDMLNet(embed_dim=32, n_qubits=4, n_layers=1, pretrained=False).to(device)
    loss_fn = AQPMLoss(n_classes=5, n_qubits=4).to(device)

    B = 10
    images = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=device)

    out = model(images)
    assert out["x"].shape == (B, 32)
    assert out["psi"].shape == (B, 16)

    loss, stats = loss_fn(out["psi"], out["x"], labels)
    assert not torch.isnan(loss)
    loss.backward()

    # Check circuit gradients exist
    assert model.vqc.weights.grad is not None


def test_cqhn_forward_backward():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QDMLNet(embed_dim=32, n_qubits=4, n_layers=1, pretrained=False).to(device)
    loss_fn = CQHNLoss().to(device)

    B = 10
    img_a = torch.randn(B, 3, 224, 224, device=device)
    img_b = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=device)

    out_a = model(img_a)
    out_b = model(img_b)

    loss, stats = loss_fn(
        psi_a=out_a["psi"],
        psi_b=out_b["psi"],
        x_a=out_a["x"],
        x_b=out_b["x"],
        z_raw_a=out_a["z_raw"],
        labels=labels,
        epoch=0,
    )
    assert not torch.isnan(loss)
    loss.backward()
    assert model.vqc.weights.grad is not None
