# Experiment protocol

Every table and figure the paper needs, with the command that produces it and
the rule that keeps it comparable to the others. Section 20 of the brief lists
twelve artefacts; this maps each one to a file on disk.

---

## Rules that apply to everything

1. **One evaluation path.** Every arm — baseline, ablation, proposed method —
   is scored by `qdml/eval/pipeline.py::evaluate`. No arm gets its own metric
   code. This is what makes a difference in a results table a difference in the
   *method* rather than in the measurement.
2. **Fixed partitions.** The same query/gallery split, the same perturbation
   parameters, the same few-shot episode seed for every arm.
3. **Validation-only model selection.** Checkpoints are chosen on validation
   mAP. The test split is touched once, at evaluation. Open-set thresholds come
   from validation.
4. **≥3 seeds** for anything in a headline table; report `mean ± std` and the
   95 % interval.
5. **Every arm reports its metric dimensionality** (`q` for quantum, 32 for
   classical). A 4-D read-out losing to a 32-D embedding is a capacity result,
   not a quantum result, and the table must let a reader see that.
6. **Both spaces, always.** Every quantum run is also evaluated in the classical
   space of the *same model* (`cls_*` columns), so "what did the circuit add" is
   a subtraction, not an argument.

---

## Table 1 — Dataset and class distribution

```bash
python scripts/02_audit_data.py --data data/processed --out results/tables
```

→ `results/tables/table1_dataset.{json,csv,md}`,
`table1b_conditions.*`, `dataset_audit.json`

Report: per-class crop counts by split, total, source-frame count, split ratio,
distinct source resolutions, the fraction of defects under 2 % of frame area,
and the leakage/duplicate verdict. The last item belongs in the paper — it tells
a reviewer the retrieval numbers are not inflated.

---

## Table 2 — Main retrieval comparison

```bash
python scripts/05_ablations.py --grid baselines --data data/processed --seeds 0,1,2
```

→ `results/grids/baselines/aggregated.{json,csv,md}`

Columns: id · description · metric dim · classical params · quantum params ·
R@1 · R@5 · R@10 · mAP · mAP@R, each `mean ± std`.

The headline comparison is AQPM/CQHN against the **strongest classical
metric-learning baseline** (B1–B5), not against B6 (softmax). Bold the best
classical arm and the best quantum arm separately so the reader can see both.

---

## Table 3 — Nearest-neighbour and few-shot

```bash
python scripts/04_evaluate.py --run results/runs/aqpm_s0 --data data/processed
```

→ `results/runs/<run>/eval/evaluation.json` → `knn`, `few_shot`

kNN at `k ∈ {1,3,5}`: Accuracy, Macro-F1, macro Precision, macro Recall.
Few-shot at `K ∈ {1,3,5}`: accuracy and Macro-F1 as `mean ± std` over 20
episodes.

Few-shot is where a prototype method should look best, and where the small
quantum read-out is least penalised — if AQPM wins anywhere, expect it here.

---

## Tables 4 and 5 — Ablations

```bash
python scripts/05_ablations.py --grid aqpm --data data/processed --seeds 0,1,2
python scripts/05_ablations.py --grid cqhn --data data/processed --seeds 0,1,2
```

→ `results/grids/{aqpm,cqhn}/aggregated.*`

Present as **Δ against the full model**, not as absolute numbers — a reader
wants "removing X costs 0.04 mAP". `scripts/07_figures.py` draws exactly that.

Group the rows:

| group | AQPM | CQHN |
|---|---|---|
| novelty removal | A1, A2, A3, A4, A5 | B1, B2, B3, B4 |
| quantum contribution | A6, A6b, A6c | B5, B6 |
| resource sensitivity | A7, A8 | B7, B8 |
| **correction checks** | A3b, A4b, A9 | B9, B10 |

The last row is this project's addition. Each entry runs the *literal* form of
something the brief or the reference code specifies, so the correction is
evidenced rather than asserted. Give it its own subsection in the paper — it is
a methods contribution.

---

## Table 6 — Robustness

```bash
python scripts/06_robustness.py \
    --runs results/runs/aqpm_s0,results/runs/cqhn_s0,results/runs/triplet_s0 \
    --data data/processed
```

→ `results/tables/table6_robustness.*`, `table6b_robustness_summary.*`

Per (method, family, severity): `ΔD⁺`, `ΔD⁻`, `robustness_margin = ΔD⁻ − ΔD⁺`,
R@1, mAP, and the fraction of the clean score retained.

Protocol: query pixels perturbed, gallery and labels untouched, triples fixed
once from clean embeddings, level 0 the identity in every family. A robust
embedding keeps `ΔD⁺` small (same defect stays close) and `ΔD⁻` non-negative
(different defects do not collapse together).

---

## Table 7 — Quantum resource analysis

```bash
python scripts/05_ablations.py --grid resources --data data/processed \
    --seeds 0,1,2 --resource-algorithm aqpm
```

→ `results/grids/resources/aggregated.*`

Columns: `q` · `L` · Hilbert dim · circuit depth · quantum params · two-qubit
gates · classical params · expectation evals · fidelity evals · train time ·
inference time · retrieval time · mAP · **RE = mAP/(q·L)**.

`RE` is always printed with its caveat string. It is a resource-normalised
indicator; the denominator `q·L` has no classical counterpart, so it cannot
compare a quantum arm to a classical one. Say this in the caption, not only in
the text.

Add the hardware column from `shot_cost()`: the number of fidelity evaluations
the protocol consumed and the shot budget that implies on a device
(compute-uncompute circuit at ≈2× encoder depth, ~10³ shots per estimate). A
simulator wall clock alone understates the real cost by orders of magnitude and
a reviewer will say so.

---

## Table 8 — Open-set recognition

→ `results/runs/<run>/eval/evaluation.json` → `open_set`

Leave-one-class-out over all five classes: AUROC, FPR@95TPR, the validation-derived
threshold, and open-set accuracy at that threshold. Report the per-class values
and the mean — a method that rejects `damaged_cover_plate` well and
`silica_gel_discoloration` badly is telling you something about the embedding.

---

## Figures

| # | file | command |
|---|---|---|
| 1 | `fig1_class_distribution` | `07_figures.py --data …` |
| 2 | `fig2_retrieval` | `--grid results/grids/baselines` |
| 3 | `fig3_fewshot` | `--grid results/grids/baselines` |
| 4 | `fig4_ablation` | `--ablation-grid results/grids/aqpm` |
| 5 | (as 4, for CQHN) | `--ablation-grid results/grids/cqhn` |
| 6 | `fig6_robustness` | `--robustness results/tables/table6_robustness.json` |
| 7 | `fig7_resources` | `--resource-grid results/grids/resources` |
| 8 | `fig8_openset` | `--run results/runs/aqpm_s0` |
| 9, 10 | architecture diagrams | TikZ source in [`06_PAPER_OUTLINE.md`](06_PAPER_OUTLINE.md) |
| 11 | `fig11_embeddings` | `--run … --projection umap` |
| 12 | `fig12_retrieval_examples`, `fig12b_xai` | `08_xai.py --run …` |
| 13 | `fig13_training` | `--run …` (training curves + quantum health) |

Figure 11 uses **identical projection settings in both panels** — same method,
same seed, same preprocessing. Comparing a UMAP of the classical embedding
against a PCA of the quantum one would be meaningless, and it is an easy mistake
to make.

Figure 12b should include at least one **failure**: a high-similarity match to
the wrong class with the attribution map showing background clutter driving it.
That single panel does more for the paper's credibility than three more
percentage points.

---

## Statistical protocol

```
μ = (1/S) Σ_s x_s        σ = sqrt( (1/(S−1)) Σ_s (x_s − μ)² )        S = seeds
95% CI ≈ μ ± 1.96 σ/√S
```

`qdml/utils/report.py::aggregate_seeds` emits `<metric>`, `<metric>_std`,
`<metric>_str` ("0.7412 +- 0.0083"), `<metric>_ci_low`, `<metric>_ci_high`.

With S = 3 the normal approximation is optimistic. Say "mean ± std over 3 seeds"
and use the interval for the advantage check, not for a significance claim. If a
gap between two arms is smaller than the seed spread, **report it as a tie**.

---

## The claim-discipline gate

```bash
python scripts/09_report.py --results results --out results/REPORT.md
```

`advantage_check` requires **all** of:

- ≥ 3 seeds with a reported interval;
- controls parameter-matched within 10 %;
- identical metric dimensionality across arms;
- the quantum lower bound above the classical upper bound.

If any fails, `advantage_claim_licensed: False` and the report prints the
blocking reasons plus the permitted wording — *resource-efficient hybrid quantum
metric learning*. Read that section before writing the abstract, and take it
seriously: it is doing the job an unsympathetic reviewer would otherwise do.

---

## Reproducibility

Every run directory contains everything needed to regenerate itself:

```
results/runs/<name>_s<seed>/
├── config.json      the fully resolved config, every field
├── best.pt          checkpoint selected on validation mAP
├── last.pt          final epoch
├── history.json     per-epoch train + validation metrics, incl. quantum health
├── summary.json     best epoch, resource report, timings, config
└── eval/
    ├── evaluation.json   every task, every space
    └── summary.{json,csv,md}
```

To reproduce a published number: `python scripts/03_train.py --config
results/runs/<name>/config.json`, then `04_evaluate.py --run <dir>`.
`tests/test_integration.py::test_training_is_seed_reproducible` asserts that a
fixed seed gives an identical validation mAP.
