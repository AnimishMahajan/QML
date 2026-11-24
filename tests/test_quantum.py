"""
Tests for Quantum Simulation, Fidelities, and Findings F1, F2, F10.
"""

import math
import pytest
import torch
import numpy as np

from qdml.quantum.statevector import (
    StateVectorSimulator,
    state_fidelity,
    pairwise_state_fidelity,
    density_matrix_from_states,
    state_to_density_fidelity,
    density_purity,
)
from qdml.quantum.circuits import VQC, CircuitConfig
from qdml.quantum.distances import compute_quantum_similarity, compute_quantum_distance


def test_statevector_simulator_basic_gates():
    sim = StateVectorSimulator(n_qubits=2)
    state = sim.zero_state(batch_size=1)

    # 1. Apply RY(pi) to wire 0 -> flips |0> to |1> on wire 0
    angles = torch.tensor([math.pi])
    state = sim.apply_ry(state, angles, wire=0)
    # Wire 0 is |1>, wire 1 is |0> -> state should be |10> (index 2 or 1 depending on convention)
    # Check expectation of Z on wire 0 is -1, wire 1 is +1
    z0 = sim.expectation_z(state, wire=0).item()
    z1 = sim.expectation_z(state, wire=1).item()
    assert pytest.approx(z0, abs=1e-5) == -1.0
    assert pytest.approx(z1, abs=1e-5) == 1.0


def test_cnot_entanglement():
    sim = StateVectorSimulator(n_qubits=2)
    state = sim.zero_state(batch_size=1)
    # Apply RY(pi/2) to wire 0 -> (|0> + |1>)/sqrt(2)
    state = sim.apply_ry(state, torch.tensor([math.pi / 2]), wire=0)
    # CNOT with control 0, target 1 -> Bell state (|00> + |11>)/sqrt(2)
    state = sim.apply_cnot(state, control=0, target=1)

    # In Bell state, <Z0> = 0, <Z1> = 0, <Z0 Z1> = 1
    z0 = sim.expectation_z(state, wire=0).item()
    z1 = sim.expectation_z(state, wire=1).item()
    zz = sim.expectation_zz(state, wire1=0, wire2=1).item()
    assert pytest.approx(z0, abs=1e-5) == 0.0
    assert pytest.approx(z1, abs=1e-5) == 0.0
    assert pytest.approx(zz, abs=1e-5) == 1.0


def test_zfid_has_the_sign_degeneracy_and_zcos_does_not():
    """Finding F2 verification: zfid treats opposite vectors z and -z as identical (distance 0)."""
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    z2 = torch.tensor([[-1.0, 0.0, 0.0]])  # exact opposite

    # Compute distances under both modes
    d_zfid = compute_quantum_distance("zfid", psi_i=None, psi_j=None, z_i=z1, z_j=z2).item()
    d_zcos = compute_quantum_distance("zcos", psi_i=None, psi_j=None, z_i=z1, z_j=z2).item()

    # zfid gives distance 0 (maximally similar!) which violates metric axioms
    assert pytest.approx(d_zfid, abs=1e-5) == 0.0
    # zcos correctly gives distance 1 (maximally distant)
    assert pytest.approx(d_zcos, abs=1e-5) == 1.0


def test_mixed_prototype_beats_mean_state_under_phase_scatter():
    """Finding F10 verification: density matrix prototype is invariant to unobservable global phase."""
    q = 2
    dim = 1 << q
    # Two states differing only by a global phase exp(i * pi) = -1
    psi1 = torch.zeros((1, dim), dtype=torch.complex64)
    psi1[0, 0] = 1.0

    psi2 = torch.zeros((1, dim), dtype=torch.complex64)
    psi2[0, 0] = -1.0  # global phase shift by pi

    states = torch.cat([psi1, psi2], dim=0)

    # 1. State vector average: (psi1 + psi2)/2 = 0 (complete cancellation!)
    mean_state = torch.mean(states, dim=0, keepdim=True)
    norm = torch.norm(mean_state).item()
    assert pytest.approx(norm, abs=1e-5) == 0.0  # cancelled out!

    # 2. Mixed state density matrix: ( |psi1><psi1| + |psi2><psi2| ) / 2
    rho = density_matrix_from_states(states)
    fid1 = state_to_density_fidelity(psi1, rho).item()
    fid2 = state_to_density_fidelity(psi2, rho).item()

    # Density matrix maintains full overlap = 1.0 without phase cancellation
    assert pytest.approx(fid1, abs=1e-5) == 1.0
    assert pytest.approx(fid2, abs=1e-5) == 1.0
    assert pytest.approx(density_purity(rho).item(), abs=1e-5) == 1.0
