# QDML for Substation Defect Recognition

**Quantum Deep Metric Learning for Robust Substation Equipment Appearance Defect Recognition.**

A hybrid classical-quantum metric learning framework designed for industrial inspection imagery. Supports:
- **AQPM-QDML**: Reliability-weighted mixed-state quantum prototypes with ambiguity-adaptive margin.
- **CQHN-QDML**: Condition-aware quantum hard-negative deep metric learning with dual symmetric condition views.
- **Pure-PyTorch Vectorized Statevector Engine**: Vectorized, differentiable simulation on CUDA tensors.

## Environment & Setup
Manage virtual environments with Miniconda on Python 3.13:
```bash
conda activate qml
pip install -e .
```

## Running the Pipeline
```bash
# 1. Environment & CUDA Doctor
python scripts/00_check_env.py

# 2. Dataset Preparation (Pascal VOC to defect crops)
python scripts/01_prepare_data.py --raw "dataset/Substation Equipment Appearance Defect Dataset/bdz-xml-6"

# 3. Dataset Audit
python scripts/02_audit_data.py

# 4. Train AQPM-QDML
python scripts/03_train.py --config configs/aqpm.yaml

# 5. Evaluate
python scripts/04_evaluate.py --run results/runs/aqpm_s0

# 6. Ablation Grids
python scripts/05_ablations.py --grid aqpm --seeds 0,1,2

# 7. Compile Report
python scripts/09_report.py
```
