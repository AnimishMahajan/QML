# Execution path

The brief's sequence is

```
dataset audit → classical metric baseline → quantum similarity baseline
    → AQPM-QDML → CQHN-QDML
```

This document turns that into dated stages with **go/no-go gates**. A gate is a
measurable condition. If a gate fails, the fix is listed — do not carry a failed
gate forward, because every later result inherits it.

Timings assume one workstation. The circuit cost is negligible after F14; wall
clock is dominated by the CNN and by image I/O.

---

## Stage 0 — Environment (half a day)

```bash
python scripts/00_check_env.py
```

### Known issue on this machine

`torch 2.11.0+cpu` is installed alongside an **RTX 5070 Ti Laptop (12 GB)**.
The GPU is unused. Fix before anything else:

```powershell
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

`cu128` or newer is required — RTX 50-series is Blackwell (sm_120) and `cu126`
wheels will not run on it. Verified available for Python 3.14 on Windows:
`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`.

Then re-run the doctor and confirm `CUDA available: True`. Expect roughly a
10–30× speedup on the CNN; at 224² this is the difference between ~25 min and
~2 min per 40-epoch run.

**Gate 0** — all of:
- [ ] `CUDA available: True`
- [ ] `fast simulator matches PennyLane (max |diff| < 1e-10)`
- [ ] `pytest tests/ -q` all green
- [ ] quantum fwd+bwd, batch 64, q=6 L=2 under 10 ms

---

## Stage 1 — Data (1–2 days, mostly waiting on access)

```bash
# while access is pending — validates everything downstream
python scripts/01_prepare_data.py --synthetic --sim-images 1200 --out data/processed_sim
python scripts/run_all.py --smoke

# once the download is in hand
python scripts/01_prepare_data.py --raw "<download root>" --out data/processed
python scripts/02_audit_data.py --data data/processed --out results/tables
```

Work through the checklist in [`03_DATASET.md §7`](03_DATASET.md) line by line.

**Gate 1** — all of:
- [ ] `leakage_check.source_images_in_multiple_splits == 0`
- [ ] all 5 classes present in train, val **and** test
- [ ] no duplicate crop group spans splits
- [ ] `class_names` verified by eye against ~10 crops per class
- [ ] split ratio ≈ 7:2:1, or the deviation is written down

*Failure → fix:* missing class in a split → re-run with the published split, or
regenerate with a larger `--min-box` if tiny boxes are eating a rare class.
Duplicates across splits → drop the duplicate group before proceeding.

---

## Stage 2 — Classical metric baseline (2–3 days)

Nothing quantum yet. Establish the number everything else is measured against.

```bash
python scripts/03_train.py --config configs/baseline_triplet.yaml --data data/processed
python scripts/04_evaluate.py --run results/runs/triplet_s0 --data data/processed
python scripts/05_ablations.py --grid baselines --data data/processed --seeds 0 \
    --only B1_contrastive,B2_triplet,B3_supcon,B4_cosface,B6_softmax
```

Tune here, not later: learning rate, `batch_k`, augmentation strength, epochs.
Whatever you settle on becomes the shared setting for every arm, including the
quantum ones — a hyperparameter advantage handed to one arm is the easiest way
to produce a result that does not replicate.

**Gate 2** — all of:
- [ ] the best classical baseline clearly beats B0 (untrained metric)
- [ ] Recall@1 comfortably above chance (chance ≈ largest class share)
- [ ] `silhouette > 0` in the classical embedding
- [ ] the ranking of the classical baselines is stable across 2 seeds

*Failure → fix:* if triplet ≈ untrained, the embedding is not learning. Check
`batch_k ≥ 4` (batch-hard mining needs several positives per class), check the
learning rate, and look at `fig11_embeddings` before touching anything quantum.

---

## Stage 3 — Quantum similarity baseline (2–3 days)

Still no novelties. Prove the quantum branch carries signal.

```bash
python scripts/05_ablations.py --grid baselines --data data/processed --seeds 0 \
    --only B7_fixed_qmap,B8_vqc_head
python scripts/05_ablations.py --grid aqpm --data data/processed --seeds 0 \
    --only A9_topq_slice,A6c_zfid_metric
```

**Gate 3 — the most important gate in the project.** All of:
- [ ] `q_z_std_mean > 0.1` throughout training (`history.json`)
- [ ] mean pairwise fidelity on held-out data in roughly `0.3–0.8`, **not** `> 0.95`
- [ ] B8 (VQC head, no novelties) within striking distance of B2 (triplet) — the
      quantum head need not win, but it must not be near-random
- [ ] `A9_topq_slice` run, with the **encoding diagnostic** recorded for both
      slices (`angle_std`, `z_std_mean`, `pairwise_fidelity_mean`). The
      diagnostic must separate them; the *retrieval* score may not, because a
      strong classical backbone can mask the whole effect — see F5. Record
      whichever you observe.

*Failure → fix:* if fidelity sits near 1, the encoding collapsed (F5). Check
`slice_mode: learned`, raise `encode_scale`, raise `lr_quantum`. If the circuit
never moves, `lr_quantum` is too low (F9). Re-read F5 and F9 before proceeding —
**everything after this stage is meaningless if the quantum branch is dead**, and
it fails silently.

---

## Stage 4 — AQPM-QDML (1 week)

```bash
python scripts/03_train.py --config configs/aqpm.yaml --data data/processed
python scripts/04_evaluate.py --run results/runs/aqpm_s0 --data data/processed
python scripts/05_ablations.py --grid aqpm --data data/processed --seeds 0
```

Watch, every epoch:

| log key | healthy | if not |
|---|---|---|
| `r_mean` | ~0.4–0.7, `r_std > 0.05` | collapse (F3) — check `normalize_reliability` |
| `classes_in_batch` | 5 | PK sampler misconfigured |
| `active_frac` | 0.2–0.8 | 0 = margin too small; 1 = margin too large |
| `q_z_std_mean` | > 0.1 | encoding collapse (F5) |
| `D_pos` vs `D_neg` | separating | the metric is not learning |

Then decide **D1** (margin sign) from `A3` / `A3b` / default, and **D3**
(read-out width) from `A7`.

**Gate 4** — all of:
- [ ] AQPM ≥ the best classical baseline, or the gap is explained and quantified
- [ ] `A1` (no reliability weighting) measurably *worse* than `A0` — otherwise
      novelty 1 is doing nothing and should not be claimed
- [ ] `A3` and `A3b` both run, and the reported margin form matches the prose
- [ ] `A6` (no quantum branch) run at matched parameters — this is the honest
      "what does the circuit buy" number
- [ ] 3 seeds, `mean ± std` reported

*Failure → fix:* if every ablation lands within noise of `A0`, the novelties are
not active. Check `r_std` (F3) and the health of the quantum branch (F5) before
concluding anything about the method.

---

## Stage 5 — CQHN-QDML (1 week)

Only after Gate 4.

```bash
python scripts/03_train.py --config configs/cqhn.yaml --data data/processed
python scripts/05_ablations.py --grid cqhn --data data/processed --seeds 0
```

Watch:

| log key | healthy | if not |
|---|---|---|
| `q_z_std_mean` | stable or rising | collapse (F7) — lengthen `cqhn_inv_warmup_epochs` |
| `valid_negatives` | = batch size | mining guard firing (F8) — check the PK sampler |
| `self_positive_frac` | ≈ 0.5 under `positive: mixed` | many singleton classes per batch |
| `L_inv` | falls, does not hit 0 | 0 means collapse |
| `ambiguity` | roughly stable | rising fast = mining is degenerate |

**Gate 5** — all of:
- [ ] `cc_gap_mAP` (within-bucket minus cross-bucket retrieval) **smaller** for
      CQHN than for AQPM and for the classical baseline — this is CQHN's whole claim
- [ ] robustness retention higher than AQPM under blur / brightness / resolution
- [ ] `B3` (no invariance loss) worse on cross-condition than `C0`
- [ ] `B9` (self-view positives, the reference behaviour) reported — evidence for F6
- [ ] 3 seeds

---

## Stage 6 — Full experimental grid (3–5 days, mostly unattended)

```bash
python scripts/05_ablations.py --grid baselines --data data/processed --seeds 0,1,2
python scripts/05_ablations.py --grid aqpm      --data data/processed --seeds 0,1,2
python scripts/05_ablations.py --grid cqhn      --data data/processed --seeds 0,1,2
python scripts/05_ablations.py --grid resources --data data/processed --seeds 0,1,2 \
    --resource-algorithm aqpm

python scripts/06_robustness.py \
    --runs results/runs/aqpm_s0,results/runs/cqhn_s0,results/runs/triplet_s0 \
    --data data/processed
```

Rough budget on a 12 GB GPU at 224², ~3,000 train crops, 40 epochs:
≈ 3–6 min per run × (11 baselines + 16 AQPM + 13 CQHN + 9 resource) × 3 seeds
≈ 7–15 hours. Run it overnight; `05_ablations.py` writes `raw_per_seed` after
every run, so an interruption costs one run, not the grid.

**Gate 6** — all of:
- [ ] `failures.json` is absent or empty in every grid
- [ ] every headline metric has `n_seeds == 3`
- [ ] the classical-arm ranking is unchanged from Stage 2 (a shifted ranking
      means something in the shared pipeline changed underneath you)

---

## Stage 7 — Figures, XAI, report (2–3 days)

```bash
python scripts/07_figures.py --data data/processed --run results/runs/aqpm_s0 \
    --grid results/grids/baselines --ablation-grid results/grids/aqpm \
    --resource-grid results/grids/resources \
    --robustness results/tables/table6_robustness.json --projection umap
python scripts/08_xai.py --run results/runs/aqpm_s0 --data data/processed --n 6
python scripts/09_report.py --results results --out results/REPORT.md
```

`09_report.py` prints the claim-discipline verdict first. Read it before writing
the abstract.

**Gate 7** — all of:
- [ ] `REPORT.md` shows the advantage verdict and its blocking reasons
- [ ] every table in [`05_EXPERIMENT_PROTOCOL.md`](05_EXPERIMENT_PROTOCOL.md) exists
- [ ] `fig12b_xai` shows at least one *failure* case — a false match driven by
      background — because that is the finding worth publishing
- [ ] the embedding figure uses identical projection settings in both panels

---

## Stage 8 — Writing (1–2 weeks)

Follow [`06_PAPER_OUTLINE.md`](06_PAPER_OUTLINE.md). Before submission:

- [ ] the dataset section states the crop protocol and per-class instance counts (F12)
- [ ] the margin form in the equations matches the one you ran (F4/D1)
- [ ] the similarity definition is stated once and used consistently (F2)
- [ ] the read-out dimensionality of every arm is in the results table (D3)
- [ ] hardware shot cost is reported, not just simulator wall clock
- [ ] no sentence claims quantum advantage unless `advantage_check` licensed it
- [ ] limitations include everything in `00_MASTER_SPEC.md §15`

---

## Critical path at a glance

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5 ──► Stage 6 ──► Stage 7 ──► Stage 8
 env        data      classical    quantum       AQPM        CQHN        grid      figures     paper
 0.5d      1-2d        2-3d         2-3d          1w          1w         3-5d        2-3d       1-2w
                                      ▲
                          the gate that silently fails
```

**Parallelisable:** Stage 0 and the synthetic run can happen while dataset
access is pending. Literature work ([`02_LITERATURE.md`](02_LITERATURE.md)) and
the architecture diagrams can be done any time.

**Not parallelisable:** Stage 3 before 4 before 5. The brief is explicit and it
is right — do not implement both algorithms at once.

---

## If you are short on time

Minimum defensible paper, in priority order:

1. Stages 0–2 — data integrity and a real classical baseline. Without this
   nothing else is interpretable.
2. Stage 3 — proof the quantum branch is alive (Gate 3).
3. Stage 4 + AQPM ablations `A0, A1, A3, A6` at 3 seeds.
4. Robustness sweep on AQPM vs. the best classical baseline.
5. `A9` and `A4b` — the two negative results from F5 and F3. These are cheap,
   they are genuinely useful to other people building hybrid models, and they
   make the methods section credible.

CQHN can then be presented as ongoing work rather than run badly. A well-executed
one-algorithm paper with honest ablations beats two half-run algorithms.
