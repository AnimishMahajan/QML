"""
Batched, differentiable, exact quantum state-vector simulation in pure PyTorch.
Supports up to q <= 8 qubits entirely on CUDA or CPU with autograd support.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateVectorSimulator:
    """
    Simulates small quantum circuits (q <= 8) over batches directly using PyTorch tensors.
    State representation: complex tensor of shape (batch_size, 2**n_qubits).
    """

    def __init__(self, n_qubits: int, device: Optional[torch.device] = None):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self.device = device or torch.device("cpu")

    def zero_state(self, batch_size: int, device: Optional[torch.device] = None, dtype=torch.complex64) -> torch.Tensor:
        """Returns |0...0> states of shape (batch_size, 2**n_qubits)."""
        dev = device or self.device
        state = torch.zeros((batch_size, self.dim), dtype=dtype, device=dev)
        state[:, 0] = 1.0 + 0.0j
        return state

    def apply_single_qubit_gate(self, state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
        """
        Applies a 2x2 unitary gate to wire in state.
        state: (batch_size, 2**n_qubits) or (batch_size, 2, 2, ...) with n_qubits axes.
        gate: (2, 2) or (batch_size, 2, 2)
        wire: 0-indexed wire (0 is most significant or least significant, consistent).
        """
        B = state.shape[0]
        # Reshape state to (B, 2, 2, ..., 2) with n_qubits dimensions
        state_tensor = state.view([B] + [2] * self.n_qubits)
        
        # We want to contract gate with axis (1 + wire)
        # Move target axis to the end, multiply, then permute back
        target_axis = 1 + wire
        axes = list(range(1, 1 + self.n_qubits))
        axes.remove(target_axis)
        axes.append(target_axis)
        perm = [0] + axes
        
        # permuted shape: (B, other_qubits..., 2)
        permuted = state_tensor.permute(perm)
        shape_before = permuted.shape
        # Flatten to (-1, 2)
        flat = permuted.reshape(-1, 2)
        
        if gate.dim() == 2:
            # gate: (2, 2). flat: (B * 2**(q-1), 2)
            # new_state = flat @ gate^T
            out = torch.matmul(flat, gate.t().to(flat.dtype))
        else:
            # gate is batched: (B, 2, 2)
            # permuted: (B, 2**(q-1), 2)
            rest_dim = 1 << (self.n_qubits - 1)
            p_reshaped = permuted.reshape(B, rest_dim, 2)
            # (B, rest_dim, 2) @ (B, 2, 2)^T -> (B, rest_dim, 2)
            out = torch.matmul(p_reshaped, gate.transpose(-1, -2).to(p_reshaped.dtype))
        
        out = out.reshape(shape_before)
        
        # Inverse permutation
        inv_perm = [0] * (1 + self.n_qubits)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        return out.permute(inv_perm).reshape(B, self.dim)

    def apply_ry(self, state: torch.Tensor, angles: torch.Tensor, wire: int) -> torch.Tensor:
        """
        RY(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        angles: (B,)
        """
        B = angles.shape[0]
        half_angles = angles * 0.5
        c = torch.cos(half_angles)
        s = torch.sin(half_angles)
        
        # Construct batched 2x2 real/complex matrix
        # [[c, -s], [s, c]]
        gate = torch.zeros((B, 2, 2), dtype=torch.complex64, device=angles.device)
        gate[:, 0, 0] = c
        gate[:, 0, 1] = -s
        gate[:, 1, 0] = s
        gate[:, 1, 1] = c
        
        return self.apply_single_qubit_gate(state, gate, wire)

    def apply_rz(self, state: torch.Tensor, angles: torch.Tensor, wire: int) -> torch.Tensor:
        """
        RZ(theta) = [[exp(-i theta/2), 0], [0, exp(i theta/2)]]
        angles: (B,)
        """
        B = angles.shape[0]
        half_angles = angles * 0.5
        cos_part = torch.cos(half_angles)
        sin_part = torch.sin(half_angles)
        
        exp_neg = torch.complex(cos_part, -sin_part)
        exp_pos = torch.complex(cos_part, sin_part)
        
        gate = torch.zeros((B, 2, 2), dtype=torch.complex64, device=angles.device)
        gate[:, 0, 0] = exp_neg
        gate[:, 1, 1] = exp_pos
        
        return self.apply_single_qubit_gate(state, gate, wire)

    def apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """
        Applies CNOT with control and target wires.
        Flips target when control is |1>.
        """
        B = state.shape[0]
        state_tensor = state.view([B] + [2] * self.n_qubits)
        
        ctrl_axis = 1 + control
        tgt_axis = 1 + target
        
        # When control == 1, swap target index 0 and 1
        # Use slicing / indexing
        idx_ctrl_1 = [slice(None)] * (1 + self.n_qubits)
        idx_ctrl_1[ctrl_axis] = 1
        
        sub_tensor = state_tensor[tuple(idx_ctrl_1)]
        # sub_tensor has shape missing ctrl_axis. tgt_axis is now shifted if tgt_axis > ctrl_axis
        sub_tgt_axis = tgt_axis - 1 if tgt_axis > ctrl_axis else tgt_axis
        
        # Flip along sub_tgt_axis
        flipped = torch.flip(sub_tensor, dims=[sub_tgt_axis])
        
        # Copy state and assign
        new_state = state_tensor.clone()
        new_state[tuple(idx_ctrl_1)] = flipped
        
        return new_state.reshape(B, self.dim)

    def encode_ry(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Angle encoding on |0...0>: |phi(v)> = (x)_k RY(pi * v_k)|0>
        angles: (B, n_qubits) in [-1, 1], scaled by pi internally.
        """
        B = angles.shape[0]
        state = self.zero_state(B, device=angles.device)
        scaled = angles * math.pi
        for k in range(self.n_qubits):
            state = self.apply_ry(state, scaled[:, k], wire=k)
        return state

    def expectation_z(self, state: torch.Tensor, wire: int) -> torch.Tensor:
        """
        Computes <psi| Z_wire |psi> for each batch item.
        Returns: (B,)
        """
        B = state.shape[0]
        state_tensor = state.view([B] + [2] * self.n_qubits)
        
        # Z has eigenvalues +1 for |0> and -1 for |1> along (1 + wire)
        prob_dist = torch.abs(state_tensor) ** 2
        
        axis = 1 + wire
        p0 = torch.sum(prob_dist.select(axis, 0), dim=tuple(range(1, self.n_qubits)))
        p1 = torch.sum(prob_dist.select(axis, 1), dim=tuple(range(1, self.n_qubits)))
        return p0 - p1

    def expectation_zz(self, state: torch.Tensor, wire1: int, wire2: int) -> torch.Tensor:
        """
        Computes <psi| Z_wire1 Z_wire2 |psi>.
        Returns: (B,)
        """
        B = state.shape[0]
        state_tensor = state.view([B] + [2] * self.n_qubits)
        prob_dist = torch.abs(state_tensor) ** 2
        
        ax1 = 1 + wire1
        ax2 = 1 + wire2
        
        # Eigenvalue is +1 when both 0 or both 1, -1 when different
        # Sum over all axes except batch
        dims_to_sum = list(range(1, 1 + self.n_qubits))
        # Keep track of signs
        # Generate meshgrid of signs
        indices = [torch.tensor([1, -1], device=state.device) for _ in range(self.n_qubits)]
        # We only need z1 and z2
        z1 = torch.ones([2] * self.n_qubits, device=state.device)
        z2 = torch.ones([2] * self.n_qubits, device=state.device)
        # select slice 1 for wire1 -> -1
        idx1 = [slice(None)] * self.n_qubits
        idx1[wire1] = 1
        z1[tuple(idx1)] = -1.0
        
        idx2 = [slice(None)] * self.n_qubits
        idx2[wire2] = 1
        z2[tuple(idx2)] = -1.0
        
        zz = (z1 * z2).unsqueeze(0)  # (1, 2, ..., 2)
        return torch.sum(prob_dist * zz, dim=tuple(range(1, 1 + self.n_qubits)))

    def all_z_expectations(self, state: torch.Tensor) -> torch.Tensor:
        """Returns [ <Z_1>, ..., <Z_q> ] of shape (B, n_qubits)."""
        zs = [self.expectation_z(state, k) for k in range(self.n_qubits)]
        return torch.stack(zs, dim=-1)

    def all_z_and_zz_expectations(self, state: torch.Tensor) -> torch.Tensor:
        """Returns [ <Z_0>, ..., <Z_{q-1}>, <Z_0 Z_1>, ..., <Z_{q-2} Z_{q-1}> ] of shape (B, 2q - 1)."""
        z_list = [self.expectation_z(state, k) for k in range(self.n_qubits)]
        zz_list = [self.expectation_zz(state, k, k + 1) for k in range(self.n_qubits - 1)]
        return torch.stack(z_list + zz_list, dim=-1)


def state_fidelity(psi_a: torch.Tensor, psi_b: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise or elementwise state fidelity F = |<psi_a | psi_b>|^2.
    If psi_a is (B, D) and psi_b is (B, D), returns (B,) elementwise.
    """
    # inner product: sum(conj(psi_a) * psi_b, dim=-1)
    inner = torch.sum(torch.conj(psi_a) * psi_b, dim=-1)
    return torch.abs(inner) ** 2


def pairwise_state_fidelity(psi_a: torch.Tensor, psi_b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes matrix of state fidelities between all pairs.
    psi_a: (N, D)
    psi_b: (M, D) (defaults to psi_a)
    Returns: (N, M)
    """
    if psi_b is None:
        psi_b = psi_a
    # Gram matrix: psi_a @ psi_b^dagger
    gram = torch.matmul(psi_a, psi_b.t().conj())  # (N, M)
    return torch.abs(gram) ** 2


def density_matrix_from_states(states: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Constructs density matrix rho = sum_i w_i |psi_i><psi_i| / sum_i w_i.
    states: (N, D)
    weights: (N,) >= 0
    Returns: (D, D)
    """
    N, D = states.shape
    if weights is None:
        weights = torch.ones(N, dtype=states.dtype, device=states.device) / N
    else:
        weights = (weights / (torch.sum(weights) + 1e-12)).to(states.dtype)
        
    # rho = sum_i w_i states[i, :, None] @ states[i, None, :].conj()
    # Vectorized: (states.t() * weights) @ states.conj()
    weighted_states = states.t() * weights  # (D, N)
    rho = torch.matmul(weighted_states, states.conj())  # (D, D)
    return rho


def state_to_density_fidelity(states: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Computes <psi_i | rho | psi_i> for each state in states.
    states: (B, D)
    rho: (D, D)
    Returns: (B,)
    """
    # <psi | rho | psi> = sum_k,l conj(psi_k) rho_kl psi_l
    # rho @ psi.t() -> (D, B)
    rho_psi = torch.matmul(rho, states.t())  # (D, B)
    # conj(states) * rho_psi.t() -> sum across D
    val = torch.sum(states.conj() * rho_psi.t(), dim=-1)
    return torch.clamp(val.real, 0.0, 1.0)


def density_distance_trace(rho_a: torch.Tensor, rho_b: torch.Tensor) -> torch.Tensor:
    """
    Computes 1 - Tr(rho_a @ rho_b) / sqrt(Tr(rho_a^2) * Tr(rho_b^2)).
    Bounded in [0, 1].
    """
    tr_ab = torch.trace(torch.matmul(rho_a, rho_b)).real
    tr_aa = torch.trace(torch.matmul(rho_a, rho_a)).real
    tr_bb = torch.trace(torch.matmul(rho_b, rho_b)).real
    denom = torch.sqrt(torch.clamp(tr_aa * tr_bb, min=1e-12))
    sim = torch.clamp(tr_ab / denom, 0.0, 1.0)
    return 1.0 - sim


def density_purity(rho: torch.Tensor) -> torch.Tensor:
    """Computes Tr(rho^2) in [1/2^q, 1]."""
    return torch.trace(torch.matmul(rho, rho)).real
