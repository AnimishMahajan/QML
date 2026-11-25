"""
Hybrid Classical-Quantum Backbone: ResNet-18 + QuantumSlice + VQC.
"""

from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from ..quantum.circuits import VQC, CircuitConfig


class QuantumSlice(nn.Module):
    """
    Projects classical embedding x in R^d to quantum rotation inputs v in [-1, 1]^q.
    F5 Fix:
      mode="learned" (default): Linear(d -> q) + BatchNorm1d + tanh (spans full [-pi, pi]).
      mode="topq" (ablation A9): tanh(x[:, :q]) (the reference slice that causes angle compression).
    """

    def __init__(self, embed_dim: int = 32, n_qubits: int = 6, mode: Literal["learned", "topq"] = "learned"):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.mode = mode

        if self.mode == "learned":
            self.proj = nn.Linear(embed_dim, n_qubits)
            self.bn = nn.BatchNorm1d(n_qubits)
        else:
            self.proj = None
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "learned":
            # If batch_size == 1 during inference/eval, avoid BatchNorm error
            if x.shape[0] > 1:
                h = self.bn(self.proj(x))
            else:
                h = self.proj(x)
            return torch.tanh(h)
        elif self.mode == "topq":
            return torch.tanh(x[:, :self.n_qubits])
        else:
            raise ValueError(f"Unknown slice mode: {self.mode}")


class ClassicalControlHead(nn.Module):
    """
    Parameter-matched classical replacement for the VQC (Ablation A6).
    Matches classical capacity to the circuit parameter count (2 * L * q).
    """

    def __init__(self, embed_dim: int = 32, out_dim: int = 6, n_params: int = 24):
        super().__init__()
        # 2-layer MLP matching parameter budget
        hidden_dim = max(out_dim, n_params // (embed_dim + out_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_raw = self.mlp(x)
        z = F.normalize(z_raw, p=2, dim=-1)
        return z, z_raw


class QDMLNet(nn.Module):
    """
    Unified Hybrid Model:
      I -> ResNet-18 -> GAP -> 32-D x -> QuantumSlice -> v -> VQC -> (z, z_raw, |psi>)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_qubits: int = 6,
        n_layers: int = 2,
        pretrained: bool = True,
        slice_mode: Literal["learned", "topq"] = "learned",
        observable_mode: Literal["z", "z_zz"] = "z",
        entangle_every_layer: bool = True,
        use_classical_head: bool = False,  # A6 ablation
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_classical_head = use_classical_head

        # 1. Classical feature extractor
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = resnet18(weights=weights)
        # Remove original fc layer
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        cnn_out_dim = base_resnet.fc.in_features  # 512

        # 2. Embedding projection head -> 32-D L2-normalized x
        self.head = nn.Linear(cnn_out_dim, embed_dim)

        # 3. Quantum interface
        if not use_classical_head:
            self.quantum_slice = QuantumSlice(embed_dim, n_qubits, mode=slice_mode)
            self.circuit_config = CircuitConfig(
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangle_every_layer=entangle_every_layer,
                observable_mode=observable_mode,
            )
            self.vqc = VQC(self.circuit_config)
            self.classical_control = None
        else:
            self.quantum_slice = None
            self.vqc = None
            n_params = 2 * n_layers * n_qubits
            out_dim = n_qubits if observable_mode == "z" else (2 * n_qubits - 1)
            self.classical_control = ClassicalControlHead(embed_dim, out_dim=out_dim, n_params=n_params)

    def extract_classical(self, images: torch.Tensor) -> torch.Tensor:
        """I -> GAP -> Linear -> L2-normalized x in R^32."""
        feat = self.backbone(images)  # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1)
        u = self.head(feat)
        x = F.normalize(u, p=2, dim=-1)
        return x

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        Returns dictionary containing:
          'x': classical embedding (B, d)
          'v': quantum rotation inputs (B, q) (if quantum)
          'z': normalized read-out (B, out_dim)
          'z_raw': un-normalized read-out (B, out_dim)
          'psi': quantum state vector (B, 2**q) (if quantum)
        """
        x = self.extract_classical(images)

        if not self.use_classical_head:
            v = self.quantum_slice(x)
            z, z_raw, psi = self.vqc(v)
            return {
                "x": x,
                "v": v,
                "z": z,
                "z_raw": z_raw,
                "psi": psi,
            }
        else:
            z, z_raw = self.classical_control(x)
            return {
                "x": x,
                "z": z,
                "z_raw": z_raw,
                "psi": None,
            }

    def get_optim_groups(self, lr_backbone: float = 3e-4, lr_quantum: float = 3e-2) -> list:
        """
        F9 Fix: Separate learning rates for pretrained CNN / projection vs randomly initialized VQC.
        """
        if self.use_classical_head:
            return [{"params": self.parameters(), "lr": lr_backbone}]

        classical_params = list(self.backbone.parameters()) + list(self.head.parameters()) + list(self.quantum_slice.parameters())
        quantum_params = list(self.vqc.parameters())
        return [
            {"params": classical_params, "lr": lr_backbone},
            {"params": quantum_params, "lr": lr_quantum},
        ]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        device: torch.device = torch.device("cpu"),
        config_dict: Optional[Dict] = None,
    ) -> "QDMLNet":
        """
        Robustly loads a QDMLNet model by auto-detecting quantum vs classical head
        and circuit dimensions directly from the checkpoint weights.
        """
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)

        cfg = config_dict or {}
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Auto-detect whether checkpoint is classical control or quantum head
        has_classical = any(k.startswith("classical_control") for k in state_dict.keys())
        has_quantum = any(k.startswith("vqc") or k.startswith("quantum_slice") for k in state_dict.keys())

        if has_classical and not has_quantum:
            use_classical_head = True
        elif has_quantum:
            use_classical_head = False
        else:
            use_classical_head = cfg.get("use_classical_head", False)

        embed_dim = cfg.get("embed_dim", 32)
        slice_mode = cfg.get("slice_mode", "learned")
        n_qubits = cfg.get("n_qubits", 6)
        n_layers = cfg.get("n_layers", 2)

        if "vqc.weights" in state_dict:
            vqc_shape = state_dict["vqc.weights"].shape
            n_layers = vqc_shape[0]
            n_qubits = vqc_shape[1]

        model = cls(
            embed_dim=embed_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            pretrained=False,
            slice_mode=slice_mode,
            use_classical_head=use_classical_head,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        return model
