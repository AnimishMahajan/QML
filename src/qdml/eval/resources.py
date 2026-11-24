"""
Resource quantification, hardware shot budgets, and claim discipline checking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ResourceProfile:
    n_qubits: int
    n_layers: int
    hilbert_dim: int
    quantum_params: int
    two_qubit_gates: int
    classical_params: int
    metric_dim: int
    training_time_sec: float = 0.0
    inference_time_ms: float = 0.0
    eval_fidelities_count: int = 0
    hardware_shot_estimate: int = 0
    mAP: float = 0.0
    resource_efficiency: float = 0.0  # RE = mAP / (q * L)


def count_circuit_resources(n_qubits: int, n_layers: int, entangle_every_layer: bool = True) -> Dict[str, int]:
    """
    Counts quantum parameters and 2-qubit CNOT gates.
    """
    quantum_params = n_layers * n_qubits * 2  # RY and RZ per wire per layer
    if entangle_every_layer:
        cnot_gates = n_layers * (n_qubits - 1)
    else:
        cnot_gates = n_qubits - 1

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "hilbert_dim": 1 << n_qubits,
        "quantum_params": quantum_params,
        "two_qubit_gates": cnot_gates,
    }


def shot_cost(n_fidelity_evals: int, shots_per_estimate: int = 1000) -> int:
    """
    Estimates the hardware shot budget required for n fidelity evaluations.
    On NISQ hardware, state overlap via compute-uncompute / SWAP test requires
    repeated projective measurements (~10^3 shots per estimate).
    """
    return int(n_fidelity_evals * shots_per_estimate)


def advantage_check(
    quantum_mean: float,
    quantum_std: float,
    quantum_seeds: int,
    quantum_dim: int,
    quantum_params: int,
    classical_mean: float,
    classical_std: float,
    classical_seeds: int,
    classical_dim: int,
    classical_params: int,
) -> Dict[str, Union[bool, List[str], str]]:
    """
    Binds Section 1 Claim Discipline:
    Refuses to license 'quantum advantage' unless all statistical and capacity controls pass.
    """
    blocking_reasons = []

    # 1. Minimum 3 seeds
    if quantum_seeds < 3 or classical_seeds < 3:
        blocking_reasons.append(
            f"Fewer than 3 seeds evaluated (quantum={quantum_seeds}, classical={classical_seeds})."
        )

    # 2. Matched metric dimensionality
    if quantum_dim != classical_dim:
        blocking_reasons.append(
            f"Metric dimensionality mismatch: quantum={quantum_dim}-D, classical={classical_dim}-D."
        )

    # 3. Parameter-matched controls (within 10%)
    param_ratio = abs(quantum_params - classical_params) / max(1, classical_params)
    if param_ratio > 0.10:
        blocking_reasons.append(
            f"Parameters not matched within 10% (ratio={param_ratio:.2%})."
        )

    # 4. Statistical significance (95% CI non-overlapping)
    q_ci_low = quantum_mean - 1.96 * (quantum_std / (quantum_seeds ** 0.5))
    c_ci_high = classical_mean + 1.96 * (classical_std / (classical_seeds ** 0.5))
    if q_ci_low <= c_ci_high:
        blocking_reasons.append(
            f"Confidence intervals overlap or classical arm exceeds quantum (quantum CI lower={q_ci_low:.4f}, classical CI upper={c_ci_high:.4f})."
        )

    licensed = len(blocking_reasons) == 0
    permitted_framing = (
        "Resource-controlled quantum advantage confirmed."
        if licensed
        else "Resource-efficient hybrid quantum metric learning; quantum-enhanced similarity representation."
    )

    return {
        "advantage_claim_licensed": licensed,
        "blocking_reasons": blocking_reasons,
        "permitted_framing": permitted_framing,
    }
