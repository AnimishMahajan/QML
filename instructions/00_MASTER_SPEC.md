# QDML for Substation Defect Recognition — unified specification

**Quantum Deep Metric Learning for Robust Substation Equipment Appearance
Defect Recognition.** This is the single normative document for the project.
It merges the supplied project brief
(`resources/provided/Quantum_Deep_Metric_Learning_Substation_Complete_Project.pdf`),
the supplied implementation framework
(`resources/provided/QDML Substation Defect Research Framework - Google Gemini.pdf`),
and the corrections in [`01_FINDINGS.md`](01_FINDINGS.md).

Where this document and a source disagree, **this document is what the code
implements** and the divergence is flagged inline with its finding ID. Nothing
has been silently changed.

---

## 1. Research question and claim discipline

> Can a shallow quantum metric space learn robust defect similarity under
> heterogeneous inspection conditions while using only a small quantum
> representation?

Two sub-questions, one per algorithm:

* **AQPM-QDML** — can reliability-aware quantum prototypes produce a better
  defect metric space?
* **CQHN-QDML** — can a quantum metric space become invariant to inspection
  conditions while separating visually confusing defects through adaptive hard
  negatives?

**Claim discipline (binding).** No claim of quantum advantage unless it survives
a resource-controlled comparison. The permitted framing is *resource-efficient
hybrid quantum metric learning* or *quantum-enhanced similarity representation*.
This is enforced in code: `qdml/eval/resources.py::advantage_check` returns
`advantage_claim_licensed: False` with explicit blocking reasons unless there
are ≥3 seeds, parameter-matched controls, matched metric dimensionality, and a
confidence interval that clears the classical arm. `scripts/09_report.py` prints
the verdict at the top of every report.

---

## 2. Target dataset

**Substation Equipment Appearance Defect Dataset for Complex Inspection
Scenarios** — IEEE DataPort, doi [`10.21227/df1t-fz35`](https://doi.org/10.21227/df1t-fz35),
Xuyang Wu, 15 May 2026, 3.59 GB, subscription required.

| property | value |
|---|---|
| frames | 4,235 real-world inspection images |
| categories | 5 |
| resolutions | 200+ distinct, 181×241 → 5184×3888 |
| split | 7:2:1 → 2,965 / 847 / 423 frames |
| task as published | **object detection** (YOLO-style; 640×640 recommended) |

Categories: `blurred_meter_reading`, `damaged_meter_housing`,
`silica_gel_discoloration`, `fractured_silica_gel_body`, `damaged_cover_plate`.

Properties that motivate metric learning: heterogeneous backgrounds, non-uniform
illumination, varying viewing angles, low-contrast and small defects, extreme
resolution spread.

> **Divergence — F12.** The brief treats this as an image-classification corpus.
> It is a detection corpus. The project therefore converts detections into
> **defect-instance crops** (one per annotated box, with a 15 % context margin),
> which makes the class label well-defined per sample and the 7:2:1 partition
> meaningful at instance level. See §4 and [`03_DATASET.md`](03_DATASET.md).

> **Divergence — F13.** All crops from one source frame stay in one split, and
> same-frame crops are excluded from a query's gallery in leave-one-out
> retrieval. Without this, Recall@1 measures duplicate detection.

---

## 3. Why metric learning

A classifier learns `I → y`. Metric learning learns `I → z` such that

  same defect ⟹ `d(z_i, z_j) → 0`,  different defect ⟹ `d(z_i, z_j) ≥ α`.

The embedding then supports image retrieval, nearest-neighbour recognition,
few-shot recognition, cross-condition retrieval and open-set rejection — all of
which a softmax classifier gives you none of. This is also the scientific
separation from the earlier AES-QDL work:

  AES-QDL: evidence selection → classification
  **QDML: similarity structure → embedding / retrieval**

---

## 4. Data pipeline

```
raw download (YOLO | VOC | COCO | folder-per-class)
    │  auto-detected layout
    ▼
qdml.data.prepare
    ├─ crop each annotated box + 15% context, square-pad, cap the long side at 384 px
    ├─ keep the published train/val/test dirs verbatim; otherwise generate 7:2:1
    │  grouped by source frame and stratified by the frame's majority class
    ├─ record per-crop condition attributes: brightness, contrast, sharpness
    │  (variance of Laplacian), defect scale fraction, source resolution
    └─ assert zero source frames span two splits
    ▼
data/processed/{manifest.csv, dataset_meta.json, crops/<split>/<class>/*.jpg}
```

The condition attributes matter: the dataset ships **no condition labels**, so
quantile-bucketing measured brightness / contrast / sharpness / scale is what
makes the brief's *cross-condition retrieval* protocol executable on real data
rather than only on synthetic perturbations.

**Batching.** `PKSampler` draws `P` classes × `K` samples (default 5×6 = 30).
Both prototype construction and hard-negative mining assume every class appears
in every batch; plain shuffling over 5 classes at batch 16 breaks that regularly
(F8).

---

## 5. Shared hybrid backbone

For image `I_i`:

```
F_i = f(I_i)                    lightweight CNN  (ResNet-18 scale)
h_i = GAP(F_i)                  R^512
u_i = W_h h_i + b_h             R^d,  d ∈ {16, 32}
x_i = u_i / (‖u_i‖₂ + ε)        L2-normalised classical embedding
v_i = tanh(BN(P_q x_i)) ∈ [-1,1]^q      q ∈ {4, 6, 8}          ← F5
|φ(v_i)⟩ = ⊗_k RY(π·v_ik)|0⟩            angle encoding
|ψ_i⟩ = U_L(θ)|φ(v_i)⟩                  L ∈ {1,2,3} layers      ← F1
z_i = [⟨Z_1⟩ … ⟨Z_q⟩]                   first-order read-out
ẑ_i = z_i / (‖z_i‖₂ + ε)
```

`U_L` = one layer of trainable `RY`/`RZ` per wire, **then** a nearest-neighbour
CNOT chain, repeated `L` times.

> **Divergence — F1.** The reference applies the CNOT chain once after all
> layers, which collapses an `L`-layer circuit to a 1-layer one. Configurable
> via `entangle_every_layer`.

> **Divergence — F5.** The reference uses `tanh` of the first `q` components of
> the unit vector `x`. Measured at `d = 32`: rotation std 0.54 rad instead of
> 2.0, read-out spread ~3× smaller, and encoded states 8–60× more mutually
> similar. `P_q` is a learned projection followed by batch normalisation, and
> `VQC.encoding_health()` reports whether the branch is carrying signal.

### Similarity

| mode | `S(i,j)` | when |
|---|---|---|
| **`fidelity`** (default) | `\|⟨ψ_i\|ψ_j⟩\|²` | true state fidelity, exact in simulation, SWAP test on hardware |
| `zcos` | `(1 + cos(ẑ_i, ẑ_j))/2` | cheap read-out surrogate, non-degenerate |
| `zfid` | `cos(ẑ_i, ẑ_j)²` | the reference's surrogate, kept only as an ablation |

`D = 1 − S`, all modes bounded in `[0,1]`.

> **Divergence — F2.** The brief defines both `ẑ` and state fidelity as "the"
> quantum similarity; the reference computes a third thing, `(ẑ·ẑ)²`, which is
> invariant under `ẑ → −ẑ` and therefore not a metric. True fidelity is the
> default; the surrogates are ablations `A6b` / `A6c` / `B5`.

### Simulation backend

`qdml/quantum/statevector.py` — batched, differentiable, exact, pure PyTorch.
At `q ≤ 8` the state is ≤ 256 complex amplitudes. Measured 105× faster than the
reference per-sample PennyLane loop (F14) and verified gate-for-gate against
PennyLane `default.qubit` to `< 1e-11` across the full configuration space.

---

## 6. Algorithm 1 — AQPM-QDML

*Adaptive Quantum Prototype-Metric Deep Learning.* **Implement this first.**

**Problem.** Learn a reliability-adaptive defect embedding organised around
quantum class prototypes, so ambiguous, low-contrast or acquisition-distorted
samples have reduced influence on the learned class geometry while genuine
defect structure stays compact and discriminative.

### Novelty 1 — reliability-weighted quantum prototypes

  `r_i = σ(w_r^T x_i + b_r)`,  `r̃_i = r_i / mean_batch(r)`  ← F3

  `ρ_c = Σ_{i: y_i=c} r̃_i |ψ_i⟩⟨ψ_i| / Σ_{i: y_i=c} r̃_i`  ← F10

  `S_ic = ⟨ψ_i|ρ_c|ψ_i⟩ ∈ [0,1]`,  `D_ic = 1 − S_ic`

> **Divergence — F10.** The brief's `p_c = Σ r_i z_i / ‖·‖` read on *states* is
> phase-sensitive: members differing by an unobservable global phase cancel.
> A density matrix is phase-invariant, is a genuine fidelity, and its EMA stays
> physical. `mean_state` and `z_vector` modes remain available (ablation `A2`).

> **Divergence — F3.** Mean-normalising `r` is mandatory: the un-normalised gate
> is minimised by `r → 0` without moving a single embedding.

Prototypes are computed **from the current batch** so gradients flow through
them; an EMA bank fills in classes absent from a batch and supplies the
evaluation-time prototype set.

### Novelty 2 — prototype-adaptive margin

  `S_i^hard = max_{c ≠ y_i} S_ic`

  `m_i = m_0 + γ · S_i^hard`     ← F4 (`margin_form: ambiguity`, default)

> **Divergence — F4.** The brief writes `m_i = m_0 + γ(1 − S^hard)` and then
> states that confusing samples should receive *stronger* pressure. Those
> disagree — `S^hard ≈ 1` means confusing, and the brief's form then gives the
> smallest margin. CQHN's sibling formula has the opposite sign and matches its
> own stated intent. Both forms are implemented (`ambiguity` / `brief`); report
> which you used.

### Novelty 3 — reliability-gated metric loss

  `D_i⁺ = D_{i,y_i}`,  `D_i⁻ = min_{c ≠ y_i} D_ic`

  `L_AQPM = (1/N) Σ_i r̃_i [D_i⁺ − D_i⁻ + m_i]_+`

### Novelty 4 — prototype compactness and separation

  `L_compact = (1/N) Σ_i r̃_i · D_i⁺`

  `L_sep = 1/(C(C−1)) Σ_{c≠k} [m_p − d(ρ_c, ρ_k)]_+`,  `d(ρ_c,ρ_k) = 1 − Tr(ρ_cρ_k)/√(Tr ρ_c Tr ρ_k)`

  `L_proto = L_compact + λ_sep L_sep`

> **Divergence.** The brief's `L_compact = (1/N) Σ ‖r_i z_i − p_{y_i}‖²` scales
> the *embedding* by reliability, which pulls unreliable samples toward the
> origin rather than down-weighting them, and stops the term being a distance.
> The default weights the distance instead; `compact_form: brief` reproduces the
> literal form.

### Total objective

  `L₁ = L_AQPM + λ_p L_proto + λ_c L_cls + λ_r (mean(r) − r*)²`

`L_cls` is the optional auxiliary cross-entropy (`λ_c = 0` by default).

### Training loop

1. Load the fixed 7:2:1 partition (instance crops).
2. Resize to 224², normalise, apply training-only mild augmentation.
3. Extract `x_i` (L2-normalised, `d = 32`).
4. Project to `v_i ∈ [−1,1]^q`.
5. Encode and run the shallow VQC → `|ψ_i⟩`.
6. Measure `ẑ_i`; estimate `r_i`; normalise to `r̃_i`.
7. Build reliability-weighted density-matrix prototypes `ρ_c`; EMA-update the bank.
8. Compute `S_ic`, `D_ic`; take `D⁺`, `D⁻`, `S^hard`.
9. Apply the reliability-gated adaptive-margin loss and the prototype terms.
10. Update classical and quantum parameters (separate LR groups, F9).
11. Validate on retrieval mAP; checkpoint on validation mAP only.

---

## 7. Algorithm 2 — CQHN-QDML

*Condition-aware Quantum Hard-Negative Deep Metric Learning.* Build only after
AQPM's quantum similarity mechanism is validated.

**Problem.** Separate defect *identity* from inspection *condition*: keep
embeddings of the same defect close when brightness, contrast, scale, blur or
resolution change, while separating visually similar but semantically different
defects through adaptive hard-negative metric learning.

### Novelty 1 — condition-disentangled dual views

Two independent draws from the **same** condition distribution
`A ~ {brightness, contrast, blur, resolution, scale, mild geometry}`:

  `I_i^(a) = A(I_i)`,  `I_i^(b) = A(I_i)`

  `L_inv = (1/N) Σ_i [1 − F(ψ_i^(a), ψ_i^(b))]`

> **Divergence.** The reference draws the two views from *different* transform
> families and applies torchvision transforms to already-batched float tensors
> (where `ColorJitter` is undefined). Asymmetric families let `L_inv` be
> minimised partly by learning "which family produced this", the opposite of
> condition invariance. Views are generated at PIL level from one symmetric
> distribution.

> **Divergence — F7.** `λ_inv` is ramped over `cqhn_inv_warmup_epochs` (5).
> At full strength from step 0 the global minimum "map everything to one state"
> dominates and the embedding collapses; this was observed, not hypothesised.

### Novelty 2 — classical-guided hard-negative mining

  `s^c_ij = x_i^T x_j`,  `j_i = argmax_{j : y_j ≠ y_i, j ≠ i} s^c_ij`

Only the mined pairs reach the quantum branch, so quantum cost is `O(B)` rather
than `O(B²)`.

> **Divergence — F8.** Rows with no valid negative are **excluded**, not
> defaulted to index 0 (which trains against a positive). The diagonal is masked
> too. `PKSampler` makes the case rare.

### Novelty 3 — negative-ambiguity-adaptive margin

  `a_i = s^c_{i,j_i}`,  `m_i = m_0 + γ · max(0, a_i − τ_a)`

  `L_HN = (1/N) Σ_i [D_q(i, i⁺) − D_q(i, j_i) + m_i]_+`

> **Divergence — F6.** `i⁺` is a *mined same-class partner* for half the batch
> and the self-view for the other half (`positive: mixed`). Reference behaviour
> (`self_view`) only ever compares an image to itself, which `L_inv` already
> covers.

### Novelty 4 — local-to-global similarity consistency

  `S^c = [s^c_ij]`,  `S^q = [1 − D_q(i,j)]`,  `Ŝ = (S − μ_S)/(σ_S + ε)`

  `L_struct = (1/B²) ‖Ŝ^c − Ŝ^q‖²_F`   (off-diagonal only)

### Read-out regulariser

  `L_reg = (1/N) Σ_i (‖z_i^raw‖₂ − 1)²`   ← F11 (on the **un-normalised** read-out)

### Total objective

  `L₂ = L_HN + λ_inv L_inv + λ_struct L_struct + λ_reg L_reg`

---

## 8. Why the two algorithms are different

| | AQPM-QDML | CQHN-QDML |
|---|---|---|
| primary idea | reliability-adaptive quantum prototypes | condition-aware hard-negative quantum metric learning |
| main challenge | ambiguous samples distort class geometry | inspection conditions create false distance |
| quantum role | sample-to-prototype fidelity | pairwise fidelity on mined pairs |
| core novelty | reliability-weighted mixed-state prototypes | ambiguity-adaptive hard negatives |
| main regulariser | prototype compactness / separation | similarity-structure consistency |
| primary use | prototype retrieval, few-shot | cross-condition retrieval, hard-pair matching |
| complexity | low | low–moderate |

---

## 9. Evaluation protocol

### Tasks
1. Image retrieval — top-K similar defect instances.
2. Nearest-neighbour recognition in the learned metric space.
3. Few-shot recognition, `K ∈ {1,3,5}`.
4. Cross-condition retrieval (measured condition buckets).
5. Open-set defect similarity / rejection.

### Retrieval protocols
* **`cross_split`** (headline) — query = test, gallery = train.
* **`leave_one_out`** — query and gallery are the same split, self and
  same-source-frame crops excluded.

Both are always computed. See decision D2 in `01_FINDINGS.md`.

### Metrics
`Recall@{1,5,10}`, `P@K`, `mAP` (full ranking), `mAP@R` (truncated at the number
of relevant items — less saturating with few classes, and the metric the DML
community adopted after *A Metric Learning Reality Check*).
Classification arms report Accuracy, Macro-F1, macro Precision and Recall.
Open-set reports AUROC and FPR@95TPR with the threshold selected **on validation
only**.

### Robustness (Section 12 of the brief)
Query pixels only; gallery and labels untouched. Level 0 is the identity in
every family so curves share an origin.

| family | levels |
|---|---|
| brightness | `I' = clip(I + b)`, `b ∈ {0, ±0.1, ±0.2, ±0.3}` |
| contrast | `I' = μ + a(I − μ)`, `a ∈ {1, 0.8, 0.6, 0.4, 1.3, 1.6}` |
| blur | `G_σ`, `σ ∈ {0, 0.5, 1, 2, 3}` |
| resolution | down-by-`r` then up, `r ∈ {1, 0.75, 0.5, 0.35, 0.25}` |
| scale | resize by `s` then crop/pad, `s ∈ {1, 0.8, 0.65, 1.25, 1.5}` |

Report `ΔD⁺`, `ΔD⁻`, `robustness_margin = ΔD⁻ − ΔD⁺`, and retrieval retained.
Anchor/positive/negative triples are fixed once from clean embeddings.

### Embedding analysis
PCA / UMAP **for visualisation only**, identical settings for the classical and
quantum panels. Quantitative geometry — intra/inter distance, separation ratio,
silhouette — comes from `eval/xai.py::embedding_geometry`.

### Explainability
Occlusion similarity attribution on a *pair*:
`ΔS_r = S_qg − S_(q\r)g`. Run against both the correct top match and the
highest incorrect match; the latter reveals background clutter driving false
similarity.

### Resources
Qubits, circuit depth, quantum parameters, two-qubit gates, classical
parameters, expectation / fidelity evaluations, training / inference / retrieval
time, and `RE = mAP / (q·L)` — **always printed with its caveat string**.
`shot_cost()` converts fidelity evaluations into the hardware shot budget they
would imply.

### Statistics
≥3 seeds where feasible; report `mean ± std` and a 95 % normal-approximation
interval. Identical query/gallery partitions and identical perturbation
parameters across all methods — enforced by routing every arm through
`qdml/eval/pipeline.py`.

---

## 10. Required baselines

| id | baseline |
|---|---|
| B0 | lightweight CNN + cosine metric (untrained metric) |
| B1 | CNN + contrastive loss |
| B2 | CNN + batch-hard triplet loss |
| B3 | CNN + supervised contrastive loss |
| B4 | CNN + large-margin cosine classifier (CosFace) |
| B5 | CNN embedding + prototype classifier |
| B6 | CNN + softmax classifier, kNN on the embedding |
| B7 | fixed quantum feature map + cosine metric (`L = 0`, circuit frozen) |
| B8 | classical embedding + shallow VQC metric head, no novelties |
| B9 | **AQPM-QDML** |
| B10 | **CQHN-QDML** |

The headline comparison is against the strongest *classical metric-learning*
baseline (expected: B2 or B3), **not** against B6. B4 is included because a
large-margin classifier is a genuinely strong embedding learner and omitting it
is the usual way a metric-learning paper overstates its gains.

---

## 11. Ablations

**AQPM** — `A0` full · `A1` no reliability weighting · `A2` class-mean prototype ·
`A3` fixed margin · `A3b` brief's margin formula (F4) · `A4` no reliability gate ·
`A4b` ungated reliability, no normalisation (F3) · `A5` no prototype separation ·
`A6` no quantum branch (parameter-matched classical head) · `A6b` `zcos` metric ·
`A6c` `zfid` metric (F2) · `A7` `q ∈ {4,8}` · `A8` `L ∈ {1,3}` ·
`A9` reference angle slice (F5).

**CQHN** — `C0` full · `B1` random negatives · `B2` fixed margin ·
`B3` no invariance loss · `B4` no structure loss · `B5` cosine instead of
fidelity · `B6` classical metric head · `B7` `q ∈ {4,8}` · `B8` `L ∈ {1,3}` ·
`B9` self-view positives only (F6) · `B10` no `λ_inv` ramp (F7).

Each ID is one named config delta in `scripts/05_ablations.py`. The `b`-suffixed
entries are the ablations *added* by this project to measure the corrections —
they are the material for an honest methods discussion.

---

## 12. Initial configuration

| component | setting | source |
|---|---|---|
| input size | 224 × 224 | brief Table 4 |
| backbone | ResNet-18 (pretrained) | brief Table 4 |
| classical embedding | 32-D, L2-normalised | brief Table 4 |
| qubits | 6 (sweep 4, 6, 8) | brief Table 4 |
| VQC depth | 2 layers (sweep 1, 2, 3) | brief Table 4 |
| encoding | `RY` angle, scale π | brief Table 4 |
| entanglement | nearest-neighbour CNOT chain, **per layer** | F1 |
| angle projection | learned `P_q` + BatchNorm + `tanh` | F5 |
| similarity | true state fidelity | F2 |
| prototypes | reliability-weighted density matrices, EMA 0.9 | F10 |
| batch | PK sampler, `P = 5`, `K = 6` | F8 |
| optimiser | Adam | brief Table 4 |
| lr (backbone / heads) | 3e-4 | — |
| lr (circuit) | 3e-2 | F9 |
| schedule | 2-epoch warm-up + cosine | — |
| epochs | 40 | — |
| `λ_inv` ramp (CQHN) | 5 epochs | F7 |
| primary metrics | Recall@1, Recall@5, mAP | brief Table 4 |
| secondary metrics | mAP@R, Macro-F1, few-shot accuracy, AUROC | brief Table 4 + D3 |

---

## 13. Implementation sequence

```
dataset audit → classical metric baseline → quantum similarity baseline
    → AQPM-QDML → CQHN-QDML
```

**Do not implement both algorithms simultaneously.** Verify the classical
embedding and the quantum distance calculation first. Week-by-week detail with
explicit go/no-go gates is in [`04_ROADMAP.md`](04_ROADMAP.md).

---

## 14. Complexity controls

1. Lightweight CNN, not a transformer.
2. 16–32-D classical embedding.
3. 4–8 qubits.
4. 1–3 VQC layers.
5. Hard-negative mining performed classically.
6. Quantum fidelity evaluated only for selected pairs during training.
7. First-order observables by default (`⟨Z_aZ_b⟩` available at no extra circuit cost).
8. No all-pairs quantum training.

Per batch: classical mining is `O(B²d)`; quantum evaluation is restricted to
mined pairs. In simulation the state-vector backend makes even all-pairs
fidelity cheap, but the *training* objective stays `O(B)` in quantum pairs so
the method remains meaningful as a hardware proposal — and `shot_cost()` reports
what it would cost there.

---

## 15. Limitations to state in the paper

1. Simulated circuits do not represent noisy hardware.
2. Quantum fidelity needs repeated circuit evaluations on a device; report the
   shot budget, not only the simulator wall clock.
3. Classical feature extraction remains the dominant computational component.
4. A quantum representation does not imply quantum advantage.
5. Synthetic robustness transformations do not cover all real inspection
   conditions — this is why measured-condition buckets are also reported.
6. Metric-learning performance depends on label quality and split integrity;
   the audit script reports both.
7. **Added (F12):** results are reported on defect-instance crops derived from
   the detection annotations, not on whole frames. State the crop protocol.
8. **Added (D3):** the quantum read-out is `q`-dimensional and the classical
   embedding 32-D; report the dimensionality of every arm.
