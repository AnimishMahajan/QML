# Findings, corrections and open decisions

This is the document to read before writing any of the paper. It records every
substantive problem found while turning the two supplied documents
(`resources/provided/`) into a working system, what was done about it, and what
still needs a decision from you.

Each finding has an ID (`F1`…`F14`). The IDs are referenced from the code and
from the ablation grid in `scripts/05_ablations.py`, so every correction is both
implemented *and* measurable — the corrected and the literal form can be run
side by side and reported.

Severity legend:

| | meaning |
|---|---|
| **BLOCKER** | leaving it as written produces a result that is wrong or meaningless |
| **MAJOR** | leaving it as written costs a large amount of performance or credibility |
| **MINOR** | a clarity or bookkeeping issue; fix it, mention it, move on |

---

## Summary table

| ID | Severity | Area | One line |
|---|---|---|---|
| F1 | MAJOR | circuit | The reference entangles once at the very end, so depth *L* buys nothing |
| F2 | **BLOCKER** | metric | Two incompatible definitions of quantum similarity; the coded one is sign-degenerate |
| F3 | **BLOCKER** | AQPM | The reliability gate can be driven to zero, zeroing the loss for free |
| F4 | **BLOCKER** | AQPM | The adaptive margin formula is inverted relative to its own stated purpose |
| F5 | **BLOCKER** | encoding | The reference angle range is ~3.7x too small; encoded states end up 8-60x more alike |
| F6 | MAJOR | CQHN | Positives are only self-views, so the hard-negative loss never compares two images |
| F7 | MAJOR | CQHN | `L_inv` at full strength from step 0 collapses the embedding |
| F8 | **BLOCKER** | CQHN | Masked `argmax` silently mines a *positive* when a row has no valid negative |
| F9 | MAJOR | training | One learning rate for a pretrained CNN and a randomly initialised circuit |
| F10 | MAJOR | AQPM | Averaging state vectors is phase-sensitive; a consistent class can cancel itself out |
| F11 | MINOR | CQHN | `L_reg` is identically zero as written |
| F12 | **BLOCKER** | data | The dataset is an **object-detection** benchmark, not a classification one |
| F13 | MAJOR | data | Multiple boxes per frame means split leakage is the default unless prevented |
| F14 | MAJOR | compute | One PennyLane `QNode` call per image makes the required experiment grid infeasible |

---

## F1 — The entangler is applied once, after all layers  ·  MAJOR  ·  circuit

**What the brief says.** `|ψ⟩ = U_L(θ)|φ(v)⟩`, `L ∈ {1,2,3}`, with a
nearest-neighbour CNOT chain as the entangling pattern — i.e. `L` repetitions of
{rotations, entangler}.

**What the reference code does.** In both `qdml_model.py` and `cqhn_qdml.py`
the CNOT loop sits *outside* the layer loop:

```python
for l in range(self.n_layers):
    for i in range(self.n_qubits):
        qml.RY(weights[l, i, 0], wires=i)
        qml.RZ(weights[l, i, 1], wires=i)

# Linear CNOT Entanglement Chain   <-- outside the loop
for i in range(self.n_qubits - 1):
    qml.CNOT(wires=[i, i + 1])
```

**Why it matters.** Consecutive single-qubit rotations on the same wire compose
into one rotation. With the entangler outside, an `L`-layer circuit is exactly
equivalent to a 1-layer circuit followed by one CNOT chain — a strictly
product-state encoding followed by one round of entanglement. Ablation A8
("change circuit depth") would then measure nothing, and any claim about circuit
expressivity would be unsupported.

**Fix.** `CircuitConfig.entangle_every_layer = True` (default) puts the
entangler inside the loop. Set it `False` to reproduce the reference exactly;
`circuit_resources()` reports the two-qubit-gate count either way, so the depth
ablation is honest.

---

## F2 — Two incompatible definitions of quantum similarity  ·  BLOCKER  ·  metric

**What the brief says.** Section 4 defines the quantum embedding as the
normalised vector of first-order observables

  `z_i = [⟨Z_1⟩, …, ⟨Z_q⟩] / ‖·‖`

and, two lines later, defines pairwise similarity as the state fidelity

  `S_q(i,j) = |⟨ψ_i|ψ_j⟩|²`.

These are different objects. `z` is a `q`-dimensional *classical read-out*;
`|ψ⟩` lives in a `2^q`-dimensional Hilbert space. For `q = 6` the first throws
away all but 6 of 64 complex amplitudes.

**What the reference code does.** It computes the *third* thing:

```python
cos_sim = torch.sum(z1 * z2, dim=-1)
fidelity = torch.square(cos_sim)      # (ẑ_i · ẑ_j)^2
return 1.0 - fidelity
```

**Why it matters.** `(ẑ_i · ẑ_j)²` is invariant under `z → −z`. Two samples
whose read-out vectors are exact opposites — maximally different in every
⟨Z_k⟩ — are scored as *maximally similar*, distance 0. During training that is
a free way to satisfy any margin: the optimiser can push a negative to the
antipode of the anchor and the hinge reports success. The metric is not a
metric (it violates `d(a,b)=0 ⟹ a=b`), so "learned defect embedding" would not
mean what the paper says it means.

**Fix.** `qdml/quantum/distances.py` makes the choice explicit, with three
named modes, and `fidelity` is the default:

| mode | `S` | notes |
|---|---|---|
| `fidelity` | `|⟨ψ_i|ψ_j⟩|²` | exact, from state vectors; what the brief specifies; a SWAP test / compute-uncompute circuit estimates it on hardware |
| `zcos` | `(1 + cos(ẑ_i, ẑ_j)) / 2` | cheap non-degenerate read-out surrogate |
| `zfid` | `cos(ẑ_i, ẑ_j)²` | the literal reference behaviour, kept so the degeneracy is *reported* rather than inherited |

Ablations `A6b`/`A6c` and `B5` run the surrogates against true fidelity, so the
paper can state what the read-out costs. `tests/test_quantum.py::test_zfid_has_the_sign_degeneracy_and_zcos_does_not`
pins the behaviour.

**Note for the hardware-cost section.** True fidelity is one inner product in
simulation but needs a compute-uncompute circuit of ≈2× the encoder depth and
~10³ shots per estimate on a device. `eval/resources.py::shot_cost` converts a
run's pair count into that budget so the paper reports device cost honestly.

---

## F3 — The reliability gate can be driven to zero  ·  BLOCKER  ·  AQPM

**What the brief says.**

  `r_i = σ(w_r^T x_i + b_r)`,  `L_AQPM = (1/N) Σ_i r_i [D_i⁺ − D_i⁻ + m_i]_+`

**Why it matters.** The hinge is non-negative, so `∂L/∂r_i ≥ 0` for every
sample, always. The cheapest descent direction available to the optimiser is
`w_r → −∞`: every `r_i → 0`, `L_AQPM → 0`, and **not one embedding has moved**.
Novelties 1 and 3 — the entire reliability story — evaporate, and the ablation
"remove reliability weighting" would show no difference because reliability was
already doing nothing.

**Fix.** `normalize_reliability=True` (default) uses

  `r̃_i = r_i / mean_batch(r)`,

so the weights always average to 1 and only their *relative* pattern is
learnable. A uniform rescaling of `r` now changes nothing, which removes the
free ride while preserving exactly the behaviour the novelty claims (down-weight
*relatively* unreliable samples). An optional anchor
`λ_r (mean(r) − r*)²` keeps the raw sigmoid off its saturating tails;
`configs/aqpm.yaml` sets `lambda_r: 0.01`.

Ablation `A4b` runs the un-normalised form so the collapse can be *shown* in the
paper — it is a genuinely interesting negative result about reliability-gated
metric losses, not just a bug fix.

`tests/test_losses.py::test_reliability_normalisation_blocks_the_collapse`
reproduces both behaviours.

---

## F4 — The adaptive margin is inverted  ·  BLOCKER  ·  AQPM

**What the brief says**, verbatim (Section 5.4):

> Define an ambiguity-aware margin `m_i = m_0 + γ(1 − S_i^hard)`, where
> `S_i^hard` is the highest similarity to an incorrect prototype. **Thus
> visually confusing samples receive stronger separation pressure.**

**The contradiction.** A visually confusing sample is one sitting close to a
*wrong* prototype, i.e. `S^hard ≈ 1`. Substituting: `m_i = m_0 + γ(1−1) = m_0`,
the **smallest** margin in the batch. An unambiguous sample (`S^hard ≈ 0`) gets
`m_0 + γ`, the largest. The formula does the exact opposite of the sentence
that explains it.

**Cross-check.** The sibling algorithm gets the sign right. CQHN Section 6.5:
`m_i = m_0 + γ·max(0, a_i − τ_a)` where `a_i` is the classical similarity to the
hardest negative — margin **grows** with ambiguity, matching its stated intent.
The two algorithms are meant to share this design idea, so AQPM's form is
almost certainly a transcription slip.

**Fix.** Three switchable forms, defaulting to the stated intent:

| `margin_form` | formula | use |
|---|---|---|
| `ambiguity` (default) | `m_0 + γ·S^hard` | matches the stated intent and CQHN's sign |
| `brief` | `m_0 + γ(1 − S^hard)` | the literal formula, run as ablation `A3b` |
| `fixed` | `m_0` | ablation `A3` |

**Decision you need to make.** There *is* a coherent reading under which the
literal formula is deliberate: `1 − S^hard` is the distance to the nearest wrong
prototype, so the rule becomes "don't waste gradient on hopeless samples" — a
soft curriculum, which is defensible. It simply is not what the accompanying
sentence claims. Run `A3` / `A3b` / default, report all three, and state in the
paper which sign you adopted and why. **Whatever you choose, do not leave the
mismatch between formula and prose in the write-up.**

---

## F5 — The encoding is crushed into a narrow angle range  ·  BLOCKER  ·  encoding

**What the reference code does.**

```python
x = F.normalize(u, p=2, dim=-1)          # unit vector in R^32
v = torch.tanh(x[:, :self.n_qubits])     # first q components
...
qml.RY(torch.pi * inputs[i], wires=i)
```

**The arithmetic.** A unit vector in `R^32` has components with RMS
`1/√32 ≈ 0.177`. `tanh` leaves them essentially unchanged at that magnitude, so
the rotation is `RY(π · 0.177) ≈ 0.56 rad` at *one sigma* and most components
are smaller. The encoded states are squeezed into a small neighbourhood of
`|0…0⟩`, the `⟨Z_k⟩` all sit near `+1`, and the *direction* of the normalised
read-out `ẑ` — which is what the metric actually uses — is carried by a small
residual.

**Measured** (`d = 32`, 256 samples, 2-layer circuit):

| slice | angle std (rad) | `<Z>` spread | mean pairwise fidelity |
|---|---|---|---|
| reference `tanh(x[:, :q])` | 0.54 | 0.18 – 0.21 | **0.33 – 0.58** |
| learned `tanh(BN(P_q x))` | 2.00 | 0.46 – 0.60 | **0.005 – 0.07** |

(`d = 32`, `q ∈ {4,6,8}`, 256 samples, 2-layer circuit; reproduce with
`notebooks/00_quickstart.py` step 3.)

The reference slice loses a factor ≈ 3 of read-out spread and leaves encoded
states **8–60× more mutually similar**. At `q = 4` its mean pairwise fidelity is
0.58: more than half of any two samples' state overlap is shared, so the circuit
is discriminating between images with what is left.

**Why it matters.** This is the most damaging finding in the list, because it
fails *silently*. Training still runs, the loss still descends (via the
classical branch), and retrieval numbers still look reasonable — the quantum
branch is simply not carrying the signal the paper attributes to it, so every
ablation labelled "quantum contribution" measures much less than it claims.
The effect worsens as `d` grows and as `q` shrinks, which is exactly the
lightweight regime the brief targets.

**Fix.** `models/backbone.py::QuantumSlice` with `mode="learned"` (default):
a trainable projection `P_q : R^d → R^q`, batch normalisation, then `tanh`, so
the angles actually span `[−π, π]`. Two diagnostics make the failure
*detectable* rather than silent:

* `VQC.encoding_health()` reports `z_mean_abs`, `z_std_mean`, `angle_std` and
  the mean/std of pairwise fidelity;
* the trainer logs `q_z_std_mean` every epoch and `collapse_guard` prints a
  warning when it falls below threshold.

`mode="topq"` reproduces the reference for ablation `A9`.

**How to verify it — and what not to expect.** The claim is established by the
*encoding diagnostic* (`angle_std`, `z_std_mean`, `pairwise_fidelity_mean`), not
by retrieval score. On a task the classical backbone can already solve, `A9` may
match or even beat the default: a 32-D pretrained embedding carries the decision
and a compressed quantum read-out simply contributes less, which early in
training can look like regularisation. Observed on the synthetic corpus at
2 epochs, `A9` scored *higher* mAP than the full model.

That is the finding, stated properly: **the encoding collapse is invisible in
the task metric**. It is measurable only by looking at the circuit. Report the
diagnostic table alongside the `A9` retrieval row, and say plainly whether the
retrieval gap materialised on the real dataset. Either answer is publishable;
asserting a gap that did not appear is not.

---

## F6 — CQHN positives are self-views only  ·  MAJOR  ·  CQHN

**What the reference code does.**

```python
z_anchor   = z_a
z_positive = z_b                  # the second view of the SAME image
z_hard_neg = z_a[hard_neg_indices]
```

**Why it matters.** `L_HN` then asks "is this image closer to *itself under a
different condition* than to a different-class image?" — a much weaker question
than the brief's own formulation `d(f(I_a), f(I_p)) ≪ d(f(I_a), f(I_n))` with
`I_p` a positive *sample*. Worse, `L_inv` already enforces exactly the
self-view term, so the two losses are partly redundant and the model is never
asked to pull two genuinely different images of the same defect together. That
is the whole point of metric learning here.

**Fix.** `CQHNConfig.positive`:

* `mixed` (default) — self-view for half the batch, a *mined hardest* same-class
  partner for the other half; singletons fall back to the self-view;
* `same_class` — always a mined partner;
* `self_view` — the reference behaviour, run as ablation `B9`.

---

## F7 — `L_inv` collapses the embedding if applied at full strength  ·  MAJOR  ·  CQHN

**Observed.** In an early smoke run the quantum read-out spread `σ(⟨Z⟩)` fell
from 0.34 to 0.11 over four epochs with `λ_inv = 1.0` from step 0.

**Why.** `L_inv = mean(1 − F(ψ_a, ψ_b))` has a global minimum at "map every
image to the same state". Early in training, when `L_HN` is producing small
gradients (the margin is not yet violated in a structured way), `L_inv`
dominates and drives exactly that degenerate solution. The collapsed solution
is not even a good optimum overall — with everything collapsed, `L_HN` sits at
its margin `m` forever — but gradient descent has no way back out.

**Fix.** `cqhn_inv_warmup_epochs` (default 5) ramps `λ_inv` linearly, letting
`L_HN` establish separation before invariance is enforced. `collapse_guard`
warns if the read-out spread falls below threshold anyway. Ablation `B10` runs
without the ramp, which is worth a line in the paper: the condition-invariance
objective has a degenerate optimum and needs scheduling.

---

## F8 — Unguarded hard-negative mining trains on positives  ·  BLOCKER  ·  CQHN

**What the reference code does.**

```python
S_c_negatives[label_mask] = -float('inf')
hard_neg_indices = torch.argmax(S_c_negatives, dim=1)
```

**Why it matters.** When a row has no valid negative — every batch member
shares the anchor's class — the row is all `−inf` and `argmax` returns **index
0**. That index is a *positive*, and the loss then actively pushes two
same-class samples apart. It is silent: no error, no NaN, just corrupted
gradient.

With 5 classes and the reference's `batch_size=16` drawn by plain shuffling,
all-one-class rows are not rare; with class imbalance they are common. The
diagonal is also not masked, so an anchor can mine *itself*.

**Fix.** Two layers:

1. `CQHNLoss._mine_negatives` masks the diagonal as well as same-class entries
   and returns a `valid` mask; invalid rows are **excluded** from `L_HN` rather
   than mislabelled. `logs["valid_negatives"]` makes the exclusion visible.
2. `data/dataset.py::PKSampler` builds class-balanced `P×K` batches so the
   situation is rare in the first place. With `P = 5, K = 6` every defect class
   is present in every batch — which prototype construction (AQPM) needs too.

`tests/test_losses.py::test_hard_negative_mining_never_selects_a_same_class_partner`
and `::test_single_class_batch_is_excluded_not_mislabelled` pin both.

---

## F9 — One learning rate for the CNN and the circuit  ·  MAJOR  ·  training

**Why it matters.** The backbone is ImageNet-pretrained and wants `~3e-4`. The
circuit weights are randomly initialised angles on a periodic loss surface and
want something like `~3e-2`. At `3e-4` the circuit barely moves in 40 epochs
and the model is classical with a decorative quantum layer; at `3e-2` the
pretrained backbone is destroyed in the first epoch.

**Fix.** `Trainer` puts `vqc.*` parameters in their own optimiser group with
`lr_quantum` (default `3e-2`, i.e. 100× the backbone rate). The warm-up and
cosine schedule apply per group. `lr_quantum` is a first-class hyperparameter
and should be swept — this is one of the highest-leverage knobs in the project.

---

## F10 — State-vector prototypes are phase-sensitive  ·  MAJOR  ·  AQPM

**What the brief says.** `p_c = Σ_{i∈c} r_i z_i / ‖Σ_{i∈c} r_i z_i‖`.

Read on the measurement vector `z` this is fine. Read on the *state* — which is
what "quantum prototype" and the fidelity-based `D_ic = 1 − |⟨ψ_i|p_c⟩|²` imply
— it is not: the global phase of `|ψ_i⟩` is physically unobservable, but a
vector average is sensitive to it. Two members of a class that differ only by
an unobservable phase partially cancel, so a perfectly consistent class can
produce a near-zero prototype.

**Fix.** `PrototypeBank` with `mode="mixed"` (default under fidelity) uses a
**density matrix**

  `ρ_c = Σ_{i∈c} w_i |ψ_i⟩⟨ψ_i|`,  `S(i,c) = ⟨ψ_i|ρ_c|ψ_i⟩ = Σ_j w_j F(ψ_i, ψ_j)`.

This is phase-invariant by construction, is a genuine fidelity bounded in
`[0,1]`, is cheap at these sizes (`5 × 256 × 256` complex ≈ 2.6 MB at `q=8`),
and its EMA stays a valid density matrix. `prototype_purity() = Tr(ρ_c²)` is a
free bonus diagnostic: how tight each defect class is in Hilbert space, 1 for a
perfectly consistent class and `1/2^q` for a maximally spread one.

`mode="mean_state"` keeps the literal reading so the effect can be measured
(ablation `A2`), and `tests/test_quantum.py::test_mixed_prototype_beats_mean_state_under_phase_scatter`
demonstrates it.

**Positioning note.** Mixed-state prototypes are not unprecedented — see
*Quantum Incremental Learning with Mixed State Prototypes* (arXiv:2608.10464).
Cite it; the novelty in AQPM is the **reliability weighting** of the mixture and
the prototype-adaptive margin, not the mixed state itself. Claiming the latter
would be easy to rebut in review.

---

## F11 — `L_reg` is identically zero  ·  MINOR  ·  CQHN

`L_reg = (1/N) Σ (‖z_i‖₂ − 1)²` is exactly 0 when `z` is L2-normalised, which
it is in both the brief's pipeline and the reference code. As written the term
contributes nothing and `λ_reg` is a no-op.

**Fix.** `QDMLNet` returns `z_raw` (the un-normalised `⟨Z⟩` vector) alongside
`z`, and `L_reg` is applied to `z_raw`. There it does something real: it keeps
the Pauli-Z expectations away from the origin, where the *direction* of `z` —
and therefore the metric — becomes numerically unstable.

---

## F12 — The dataset is an object-detection benchmark  ·  BLOCKER  ·  data

**What the brief assumes.** "4,235 real-world inspection images from five
appearance-defect categories", with the objective written over whole images
`I → z`.

**What the dataset is.** *Substation Equipment Appearance Defect Dataset for
Complex Inspection Scenarios*, IEEE DataPort, doi `10.21227/df1t-fz35`,
Xuyang Wu, 15 May 2026, 3.59 GB, **subscription required**. The listing
recommends 640×640 training size and mosaic augmentation and describes
YOLO-based usage — i.e. bounding-box annotations, multiple annotated defects per
frame, possibly of different classes, in a large field of substation background.

**Why it matters.** Feeding whole 5184×3888 frames to a 4–8 qubit embedding
asks it to represent *"a photo that contains a defect somewhere"*. The defect
often occupies under 2% of the frame. The learning signal is dominated by
background, and the "five categories" label is ill-defined for any frame that
contains two different defect types.

**Fix.** `qdml/data/prepare.py` converts detections into **defect instances**:
one crop per annotated box with a configurable context margin (`--context 0.15`).
This is the standard construction for retrieval / re-identification corpora, it
makes the class label well-defined per sample, and it is what makes the 7:2:1
partition meaningful at instance level. The layout detector handles YOLO, Pascal
VOC, COCO and folder-per-class so the converter works whatever the download
contains.

**This must be stated explicitly in the paper's dataset section**, with the crop
count, the context margin, and the per-class instance counts from
`scripts/02_audit_data.py`. A reviewer who knows the dataset will otherwise ask.

---

## F13 — Multiple boxes per frame make split leakage the default  ·  MAJOR  ·  data

Once one frame yields several crops, a naive random split puts crops of the
*same photograph* — often the same physical defect at near-identical scale and
lighting — into both the gallery and the query set. Recall@1 then measures
duplicate detection, not defect similarity, and every method looks better than
it is. This is the classic way a retrieval benchmark silently inflates.

**Fix, three layers:**

1. `prepare()` uses the download's published `train`/`val`/`test` directories
   verbatim when they exist, and records in `dataset_meta.json` which happened.
2. When it must generate a split, assignment is **grouped by source image** and
   stratified by the frame's majority class.
3. `_leakage_check()` asserts zero frames span splits, `scripts/02_audit_data.py`
   re-checks it and also hashes every crop for exact duplicates, and
   `retrieval_metrics(..., query_sources=, gallery_sources=)` excludes same-frame
   crops from a query's gallery in the leave-one-out protocol.

---

## F14 — One QNode call per image makes the experiment grid infeasible  ·  MAJOR  ·  compute

**What the reference code does.**

```python
for i in range(batch_size):
    z_i = self.qnode(x_q[i], self.vqc_weights)
```

**Measured on this machine** (CPU, batch 64, `q=6`, `L=2`, forward + backward):

| | per step | 40 epochs × ~3,000 crops |
|---|---|---|
| PennyLane, per-sample loop | 727 ms | ≈ 22 min of circuit time |
| batched PyTorch state vector | 6.9 ms | ≈ 0.2 min |
| | **105× faster** | |

That 22 minutes is *only the circuit*, for *one* run. The study specified in
the brief needs 2 algorithms × ~8 ablations × 3 seeds, plus a 3×3 resource
sweep and the baselines — on the order of 60–80 runs. The difference is between
an afternoon and a fortnight.

**Fix.** `qdml/quantum/statevector.py`: a batched, differentiable state-vector
simulator in pure PyTorch. At `q ≤ 8` the state is at most 256 complex
amplitudes, so simulating it directly as a tensor is exact and trivially
vectorised. It also gives true fidelity for free (F2) and runs on GPU.

**It is verified, not assumed.** `tests/test_quantum.py::test_matches_pennylane`
checks it gate for gate against PennyLane `default.qubit` across the whole
configuration space the study uses (qubits × layers × entangler × rotation set ×
encoding × re-uploading). Agreement is to `< 1e-11`; the full 648-configuration
sweep run during development agreed to `7.3e-16`. PennyLane stays in
`requirements.txt` purely as the oracle.

---

# Open decisions for you

These are genuine choices, not bugs. The code supports both sides of each.

### D1 — Margin sign in AQPM (from F4)
Adopt the stated intent (`ambiguity`, current default) or the literal formula
(`brief`)? Run `A3`, `A3b` and the default; report all three; make the prose and
the formula agree either way.

### D2 — Retrieval protocol
`cross_split` (query = test, gallery = train) is the deployment-shaped question
and is the current headline. `leave_one_out` is what most DML papers report.
Both are computed. Pick one as primary and put the other in an appendix — do
not switch between them per table.

### D3 — Read-out width
First-order `⟨Z_k⟩` gives a `q`-dimensional read-out; at `q = 4` that is a 4-D
embedding competing against a 32-D classical one. The comparison is not
capacity-matched and the classical arm may win for that reason alone.
`observables: z_zz` adds nearest-neighbour `⟨Z_aZ_b⟩` terms for `2q−1`
dimensions at no extra circuit cost. **Recommendation:** report `q ∈ {6,8}` as
the headline and include a capacity-matched control; state the dimensionality
of every arm in the results table.

### D4 — How hard to push for "quantum advantage"
The brief is already disciplined about this and the code enforces it:
`eval/resources.py::advantage_check` refuses to license the claim without
≥3 seeds, parameter-matched controls, matched metric dimensionality and a
confidence interval that clears the classical arm. `scripts/09_report.py` prints
the verdict and the blocking reasons at the top of every report.
**Recommendation:** frame the paper as *resource-efficient hybrid quantum metric
learning*, exactly as Section 26 of the brief advises. The interesting result is
the geometry — mixed-state prototypes, condition invariance, class purity in
Hilbert space — not a horse race a 6-qubit simulator cannot win.

### D5 — What to do while dataset access is pending
`SubstationSim` (`qdml/data/synthetic.py`) renders a structural surrogate and
writes it in YOLO layout, so the real download changes exactly one command-line
argument. Every component is testable and timeable today.
**Its numbers are pipeline validation and must never appear as results about the
real dataset.** Every generated corpus carries a `SYNTHETIC.txt` saying so.
