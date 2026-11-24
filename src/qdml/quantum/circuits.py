"""
Parameterised Quantum Circuits (VQC) and observable read-out modules.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
import torch
import torch.nn as nn
from .statevector import (
    StateVectorSimulator,
    pairwise_state_fidelity,
)


@dataclass
class CircuitConfig:
    n_qubits: int = 6
    n_layers: int = 2
    entangle_every_layer: bool = True  # F1: inside layer loop
    observable_mode: Literal["z", "z_zz"] = "z"
    device: str = "cpu"


class VQC(nn.Module):
    """
    Shallow Variational Quantum Circuit (VQC) implemented via batched StateVectorSimulator.
    """

    def __init__(self, config: Optional[CircuitConfig] = None, n_qubits: int = 6, n_layers: int = 2, entangle_every_layer: bool = True):
        super().__init__()
        if config is not None:
            self.n_qubits = config.n_qubits
            self.n_layers = config.n_layers
            self.entangle_every_layer = config.entangle_every_layer
            self.observable_mode = config.observable_mode
        else:
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.entangle_every_layer = entangle_every_layer
            self.observable_mode = "z"

        # Circuit weights: [n_layers, n_qubits, 2] for RY and RZ rotations
        # Initialized uniformly in [-pi, pi]
        self.weights = nn.Parameter(
            torch.empty(self.n_layers, self.n_qubits, 2).uniform_(-math.pi, math.pi)
        )
        self.sim = StateVectorSimulator(self.n_qubits)

    def forward_state(self, v: torch.Tensor) -> torch.Tensor:
        """
        Encodes inputs v in [-1, 1]^q and runs the parameterised circuit.
        Returns: state vector |psi> of shape (B, 2**n_qubits).
        """
        B = v.shape[0]
        state = self.sim.encode_ry(v)

        for l in range(self.n_layers):
            # 1. Single qubit rotations RY and RZ
            for k in range(self.n_qubits):
                ry_angle = self.weights[l, k, 0].expand(B)
                state = self.sim.apply_ry(state, ry_angle, wire=k)
                rz_angle = self.weights[l, k, 1].expand(B)
                state = self.sim.apply_rz(state, rz_angle, wire=k)

            # 2. Linear CNOT Entanglement Chain (F1: inside layer loop if entangle_every_layer)
            if self.entangle_every_layer:
                for k in range(self.n_qubits - 1):
                    state = self.sim.apply_cnot(state, control=k, target=k + 1)

        # If not entangling every layer (F1 ablation), apply chain once after all layers
        if not self.entangle_every_layer:
            for k in range(self.n_qubits - 1):
                state = self.sim.apply_cnot(state, control=k, target=k + 1)

        return state

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs VQC on batch v.
        Returns:
            z: L2-normalized expectation vector (B, q) or (B, 2q-1)
            z_raw: un-normalized expectation vector (B, q) or (B, 2q-1) (for L_reg, F11)
            state: complex state vector (B, 2**q)
        """
        state = self.forward_state(v)

        if self.observable_mode == "z_zz":
            z_raw = self.sim.all_z_and_zz_expectations(state)
        else:
            z_raw = self.sim.all_z_expectations(state)

        norm = torch.norm(z_raw, p=2, dim=-1, keepdim=True) + 1e-12
        z = z_raw / norm
        return z, z_raw, state

    def encoding_health(self, v: torch.Tensor) -> Dict[str, float]:
        """
        Diagnostic for F5: reports angle standard deviation, read-out spread,
        and pairwise fidelity mean/std across batch v.
        """
        with torch.no_grad():
            z, z_raw, state = self.forward(v)
            angles = v * math.pi
            fids = pairwise_state_fidelity(state)
            # exclude diagonal
            N = fids.shape[0]
            if N > 1:
                mask = ~torch.eye(N, dtype=torch.bool, device=fids.device)
                off_diag_fids = fids[mask]
                mean_fid = off_diag_fids.mean().item()
                std_fid = off_diag_fids.std().item()
            else:
                mean_fid = 1.0
                std_fid = 0.0

            return {
                "angle_std": angles.std().item(),
                "z_mean_abs": torch.abs(z_raw).mean().item(),
                "z_std_mean": z_raw.std(dim=0).mean().item(),
                "pairwise_fidelity_mean": mean_fid,
                "pairwise_fidelity_std": std_fid,
            }
