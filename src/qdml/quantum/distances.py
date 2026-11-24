"""
Similarity and distance computation across quantum states and read-out surrogates (F2).
"""

from __future__ import annotations
from typing import Literal, Tuple, Union, Optional
import torch
import torch.nn.functional as F
from .statevector import state_fidelity, pairwise_state_fidelity


def compute_quantum_similarity(
    mode: Literal["fidelity", "zcos", "zfid"],
    psi_i: torch.Tensor,
    psi_j: torch.Tensor,
    z_i: Optional[torch.Tensor] = None,
    z_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes pairwise or elementwise similarity S in [0, 1].
    mode:
      - 'fidelity': true quantum state fidelity |<psi_i | psi_j>|^2
      - 'zcos': (1 + cos(z_i, z_j)) / 2 (non-degenerate read-out surrogate)
      - 'zfid': cos(z_i, z_j)^2 (the reference surrogate, sign-degenerate F2)
    """
    if mode == "fidelity":
        return state_fidelity(psi_i, psi_j)

    if z_i is None or z_j is None:
        raise ValueError(f"Mode {mode} requires normalized read-out vectors z_i and z_j")

    cos_sim = torch.sum(z_i * z_j, dim=-1)

    if mode == "zcos":
        return (1.0 + cos_sim) * 0.5
    elif mode == "zfid":
        return torch.square(cos_sim)
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")


def compute_quantum_distance(
    mode: Literal["fidelity", "zcos", "zfid"],
    psi_i: torch.Tensor,
    psi_j: torch.Tensor,
    z_i: Optional[torch.Tensor] = None,
    z_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes distance D = 1 - S in [0, 1]."""
    sim = compute_quantum_similarity(mode, psi_i, psi_j, z_i, z_j)
    return 1.0 - sim


def pairwise_quantum_similarity_matrix(
    mode: Literal["fidelity", "zcos", "zfid"],
    psi_a: torch.Tensor,
    psi_b: Optional[torch.Tensor] = None,
    z_a: Optional[torch.Tensor] = None,
    z_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes full (N, M) similarity matrix between all pairs in batch.
    """
    if mode == "fidelity":
        return pairwise_state_fidelity(psi_a, psi_b)

    if z_a is None:
        raise ValueError("z_a required for read-out surrogates")
    if z_b is None:
        z_b = z_a

    cos_sim = torch.matmul(z_a, z_b.t())

    if mode == "zcos":
        return (1.0 + cos_sim) * 0.5
    elif mode == "zfid":
        return torch.square(cos_sim)
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")
