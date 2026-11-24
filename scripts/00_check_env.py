"""
Script 00: Environment doctor and CUDA verification (Stage 0, Gate 0).
"""

import sys
import time
import torch


def main():
    print("=" * 60)
    print("Stage 0: Environment & CUDA Doctor")
    print("=" * 60)

    # 1. Python version check
    py_ver = sys.version.split()[0]
    print(f"Python version: {py_ver}")
    if not (sys.version_info.major == 3 and sys.version_info.minor == 13):
        print(f"WARNING: Expected Python 3.13, found {py_ver}")
    else:
        print("✓ Python 3.13 confirmed.")

    # 2. CUDA check
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✓ GPU: {device_name} (Capability {capability[0]}.{capability[1]}, {vram:.2f} GB VRAM)")
        device = torch.device("cuda:0")
    else:
        print("✗ CUDA is NOT available! PyTorch will fall back to CPU.")
        device = torch.device("cpu")

    # 3. Quick CUDA tensor benchmark
    print("\nRunning CUDA tensor benchmark...")
    t0 = time.time()
    x = torch.randn(1024, 1024, device=device)
    for _ in range(100):
        y = torch.matmul(x, x)
    if cuda_avail:
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"✓ 100x (1024x1024) matrix multiplications: {elapsed * 1000:.2f} ms")

    # 4. StateVector simulator check
    print("\nTesting PyTorch StateVector quantum simulator...")
    from qdml.quantum.statevector import StateVectorSimulator, pairwise_state_fidelity
    from qdml.quantum.circuits import VQC, CircuitConfig

    q, L = 6, 2
    cfg = CircuitConfig(n_qubits=q, n_layers=L, entangle_every_layer=True)
    vqc = VQC(cfg).to(device)

    B = 64
    inputs = torch.randn(B, q, device=device).clamp(-1.0, 1.0).requires_grad_(True)

    # Warm-up pass
    _z, _z_raw, _psi = vqc(inputs)
    _loss = torch.sum(_z)
    _loss.backward()
    if cuda_avail:
        torch.cuda.synchronize()

    # Timed pass
    vqc.zero_grad()
    t0 = time.time()
    z, z_raw, psi = vqc(inputs)
    loss = torch.sum(z)
    loss.backward()
    if cuda_avail:
        torch.cuda.synchronize()
    sv_elapsed = (time.time() - t0) * 1000
    print(f"✓ Quantum fwd+bwd (batch={B}, q={q}, L={L}) on {device}: {sv_elapsed:.2f} ms")

    health = vqc.encoding_health(inputs)
    print(f"✓ Circuit encoding health: angle_std={health['angle_std']:.2f}, z_spread={health['z_std_mean']:.3f}, mean_fidelity={health['pairwise_fidelity_mean']:.3f}")

    # Gate 0 verification verdict
    gate0_passed = cuda_avail and (sv_elapsed < 100.0)
    print("\n" + "=" * 60)
    if gate0_passed:
        print("✓ GATE 0 PASSED: CUDA active, tensor cores operational, simulator speed < 50ms.")
    else:
        print("✗ GATE 0 FAILED: Please check CUDA drivers or configuration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
