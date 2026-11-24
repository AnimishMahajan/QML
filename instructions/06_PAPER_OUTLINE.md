# Paper outline

Structure, section by section, with the artefact that supports each claim. Also
contains the TikZ source for the two architecture figures, which are vector
drawings rather than script output.

Target venues, in rough order of fit:
*Quantum Machine Intelligence* · *Expert Systems with Applications* (where QUSL
appeared) · *IEEE Transactions on Instrumentation and Measurement* ·
*Entropy* / *Symmetry* (MDPI, fast, and where several supplied references sit).

---

## Title

> Quantum Deep Metric Learning for Robust Substation Equipment Appearance Defect
> Recognition

Subtitle option if the paper leads with the geometry rather than the
application: *"Reliability-weighted quantum prototypes and condition-invariant
hard-negative learning"*.

---

## Abstract (~200 words)

One sentence each:

1. Substation inspection imagery varies enormously in illumination, scale,
   resolution and background; defects are small and low-contrast.
2. Existing work on this data is detection and classification; neither supports
   retrieval, few-shot or open-set operation, which is what inspection teams
   actually need.
3. We formulate the problem as metric learning and present two lightweight
   hybrid quantum algorithms: AQPM-QDML (reliability-weighted quantum prototypes
   with a prototype-adaptive margin) and CQHN-QDML (condition-invariant
   embeddings with classically-guided, ambiguity-adaptive hard negatives).
4. Both use 4–8 qubits and 1–3 circuit layers, and are evaluated on defect
   instances from the Substation Equipment Appearance Defect Dataset.
5. Results, stated plainly, including where the classical baseline wins.
6. **Explicitly:** we do not claim quantum advantage; we report a
   resource-controlled comparison and characterise four failure modes of hybrid
   quantum metric models that are silent at training time.

Do not put a percentage improvement in sentence 1. Reviewers of quantum ML
papers are primed for overclaiming, and leading with the honest framing is worth
more than the number.

---

## 1. Introduction

* Substation inspection at scale; what an appearance defect is and why it matters.
* Current practice is detection/classification — cite the YOLOv8/YOLOv11/PBZGNet
  line from [`02_LITERATURE.md §2.3`](02_LITERATURE.md). One sharp paragraph:
  a detector cannot answer *"find me other instances that look like this"*.
* Why metric learning: high intra-class and low inter-class variance
  (Mohan et al.); retrieval, few-shot, cross-condition and open-set all fall out
  of one embedding.
* Why quantum, stated carefully: not speed, but a compact non-linear similarity
  structure — 6 qubits carry a 64-dimensional Hilbert space from 12 trainable
  parameters. Frame it as *representation*, not acceleration.
* Contributions (4–5 bullets, and only claim what `02_LITERATURE.md §3` says is
  defensible).

## 2. Related work

Four paragraphs: quantum metric learning (Lloyd et al.; Kim & Bekiranov;
Haug & Kim) · quantum similarity and retrieval (QUSL; QAML; the triplet-PQC
line) · classical deep metric learning (Kaya & Bilge; Mohan et al.) ·
substation defect vision.

**Say explicitly what is prior art.** Mixed-state prototypes → arXiv:2608.10464.
Perturbed positives → QUSL. Quantum triplet loss → QAML. A reviewer who finds
these unacknowledged will assume the rest of the paper is equally careless.

## 3. Problem formulation

`I → z` with `d(z_i, z_j) → 0` for same defect and `≥ α` otherwise. Retrieval,
kNN, few-shot, cross-condition and open-set as consequences. → spec §3.

## 4. Method

### 4.1 Hybrid backbone
CNN → GAP → 32-D L2-normalised `x` → learned projection + BatchNorm + `tanh` →
`RY` angle encoding → `L` layers of {`RY`,`RZ`, CNOT chain} → `|ψ⟩`. → spec §5.

**State the similarity definition once, here, and use only it thereafter:**
`S(i,j) = |⟨ψ_i|ψ_j⟩|²`, `D = 1 − S`. Note that on hardware this is a SWAP test
or compute-uncompute circuit, and give the shot cost. → F2.

State the angle range explicitly and why it is not the naive slice. → F5.

### 4.2 AQPM-QDML
Novelties 1–4, with the density-matrix prototype (F10) and the mean-normalised
reliability (F3) presented as the method, with a footnote to the ablation that
shows why. Give the margin form you actually ran and make the prose agree with
the equation (F4/D1).

### 4.3 CQHN-QDML
Novelties 1–4, with the mixed positive sampling (F6) and the `λ_inv` ramp (F7).

### 4.4 Complexity
`O(B²d)` classical mining, `O(B)` quantum pairs. Table of circuit resources.

## 5. Experimental setup

* **Dataset, with the crop protocol stated plainly.** "The dataset is published
  as an object-detection benchmark; we derive *N* defect-instance crops from the
  annotated boxes with a 15 % context margin, keeping all crops from one frame in
  one split." Give *N*, per-class counts, and the leakage/duplicate verdict. → F12, F13.
* Implementation: PyTorch, a custom batched state-vector simulator verified
  against PennyLane to `< 1e-11`, hardware, wall clock.
* Baselines (all ten), hyperparameters, seeds.
* Metrics and protocols; say which retrieval protocol is primary (D2).

## 6. Results

6.1 Main retrieval comparison (Table 2, Fig 2) — **including where classical wins**
6.2 Nearest-neighbour and few-shot (Table 3, Fig 3)
6.3 Cross-condition retrieval (the within/cross gap) — CQHN's core claim
6.4 Robustness (Table 6, Fig 6)
6.5 Open-set (Table 8, Fig 8)
6.6 Ablations (Tables 4–5, Figs 4–5)
6.7 Quantum resource analysis (Table 7, Fig 7), with the hardware shot budget
6.8 Embedding geometry (Fig 11) and prototype purity `Tr(ρ_c²)`
6.9 Explainability (Fig 12, 12b) — **include a failure case**

## 7. Discussion

### 7.1 Failure modes of hybrid quantum metric models
The strongest section in the paper if written well, and it is already measured:

* **Encoding collapse** — slicing an L2-normalised `d`-D embedding into `q`
  angles compresses rotations by ~`√d`; encoded states end up 8–60× more
  mutually similar. Evidence: the `pairwise_fidelity_mean` diagnostic, with
  ablation `A9` for the downstream effect. Note honestly whether the retrieval
  score moved — a strong classical backbone can hide the whole thing, which is
  the point. (F5)
* **Reliability collapse** — a non-negative loss gated by a learned weight is
  minimised by sending the weight to zero. Evidence: `A4b`. (F3)
* **Invariance collapse** — a condition-invariance objective has a degenerate
  global optimum at "one state for everything". Evidence: `B10`. (F7)
* **Degenerate similarity** — `(ẑ·ẑ)²` is sign-invariant and therefore not a
  metric. Evidence: `A6c`. (F2)

Each is silent: training converges, losses fall, retrieval numbers look
plausible. Give the diagnostics that detect each one. This is the part other
groups will actually use.

### 7.2 What the quantum branch buys
The `A6` / `B6` subtraction at matched parameters, plus the geometry comparison.
If the classical arm wins, say so in the first sentence of the subsection and
then discuss why — read-out dimensionality (D3), the Kim & Bekiranov
parameter-to-sample condition, simulator vs hardware.

### 7.3 Claim discipline
Reproduce the `advantage_check` verdict and its blocking reasons. Stating the
conditions under which the claim *would* be licensed is more useful than
asserting it.

## 8. Limitations
Spec §15, all eight items.

## 9. Conclusion and future work
Deeper circuits · second-order observables (D3) · hardware validation with noise
· DQFIM-guided ansatz selection (Haug & Kim) · extending the condition-invariance
objective to real temporal inspection sequences rather than synthetic views.

---

## Figure 9 — AQPM-QDML architecture (TikZ)

```latex
\begin{figure}[t]\centering
\begin{tikzpicture}[
  font=\small, node distance=6mm,
  blk/.style={draw, rounded corners=2pt, minimum height=8mm, align=center,
              inner xsep=3pt, fill=blue!5},
  qblk/.style={blk, fill=purple!8},
  lblk/.style={blk, fill=orange!10},
  ->/.default, arr/.style={-latex, thick}]

\node[blk] (img)  {Substation\\image};
\node[blk, right=of img] (pre)  {Preprocess\\crop $224^2$};
\node[blk, right=of pre] (cnn)  {Lightweight\\CNN + GAP};
\node[blk, right=of cnn] (emb)  {Compact\\embedding $x\in\mathbb{R}^{32}$};

\node[qblk, below=10mm of emb] (proj) {Learned projection\\$v=\tanh(\mathrm{BN}(P_q x))$};
\node[qblk, left=of proj]  (enc)  {Angle encoding\\$R_y(\pi v_k)$};
\node[qblk, left=of enc]   (vqc)  {Shallow VQC\\$U_L(\theta)$, $L$ layers};
\node[qblk, left=of vqc]   (st)   {Quantum state\\$|\psi_i\rangle$};

\node[lblk, below=10mm of st]   (rel)   {Reliability\\$\tilde r_i=r_i/\bar r$};
\node[lblk, right=of rel]       (proto) {Reliability-weighted\\prototypes $\rho_c$};
\node[lblk, right=of proto]     (dist)  {Fidelity distance\\$D_{ic}=1-\langle\psi_i|\rho_c|\psi_i\rangle$};
\node[lblk, right=of dist]      (loss)  {Adaptive-margin\\metric loss};
\node[lblk, below=8mm of dist]  (reg)   {Prototype compactness\\+ separation};

\draw[arr] (img)--(pre); \draw[arr] (pre)--(cnn); \draw[arr] (cnn)--(emb);
\draw[arr] (emb)--(proj); \draw[arr] (proj)--(enc); \draw[arr] (enc)--(vqc);
\draw[arr] (vqc)--(st);   \draw[arr] (st)--(rel);
\draw[arr] (rel)--(proto);\draw[arr] (proto)--(dist);\draw[arr] (dist)--(loss);
\draw[arr] (proto) |- (reg); \draw[arr] (reg) -- (loss);
\draw[arr] (emb.east) -- ++(6mm,0) |- (rel.east);

\begin{scope}[on background layer]
\node[draw, dashed, gray, rounded corners, fit=(proj)(enc)(vqc)(st),
      label={[gray]above:Quantum metric representation}] {};
\node[draw, dashed, gray, rounded corners, fit=(rel)(proto)(dist)(loss)(reg),
      label={[gray]below:Adaptive prototype metric learning}] {};
\end{scope}
\end{tikzpicture}
\caption{AQPM-QDML. Reliability $\tilde r_i$ weights both prototype
construction and the metric loss; prototypes are mixed states $\rho_c$, so the
similarity $\langle\psi_i|\rho_c|\psi_i\rangle$ is invariant to each sample's
unobservable global phase.}
\label{fig:aqpm}
\end{figure}
```

Requires `\usetikzlibrary{fit,backgrounds,positioning}`.

## Figure 10 — CQHN-QDML architecture (TikZ)

```latex
\begin{figure}[t]\centering
\begin{tikzpicture}[
  font=\small, node distance=6mm,
  blk/.style={draw, rounded corners=2pt, minimum height=8mm, align=center,
              inner xsep=3pt, fill=blue!5},
  qblk/.style={blk, fill=purple!8},
  lblk/.style={blk, fill=orange!10},
  arr/.style={-latex, thick}]

\node[blk] (img)  {Input\\crop};
\node[blk, right=of img] (views){Two condition views\\$I^{(a)},I^{(b)}\sim\mathcal{A}$};
\node[blk, right=of views](cnn) {Shared CNN\\$\to x^{(a)},x^{(b)}$};
\node[blk, right=of cnn]  (simc){Classical similarity\\$S^c=x x^\top$};
\node[blk, below=9mm of simc](mine){Hard-negative mining\\$j_i=\arg\max_{y_j\ne y_i}S^c_{ij}$};

\node[qblk, left=of mine] (trip) {Anchor / positive /\\hard negative};
\node[qblk, left=of trip] (vqc)  {Shallow VQC};
\node[qblk, left=of vqc]  (fid)  {Fidelity distance\\$D_q=1-|\langle\psi|\phi\rangle|^2$};

\node[lblk, below=9mm of fid]  (inv)  {Condition invariance\\$\mathcal{L}_{\mathrm{inv}}$};
\node[lblk, right=of inv]      (marg) {Ambiguity-adaptive\\margin $m_i$};
\node[lblk, right=of marg]     (hn)   {Hard-negative loss\\$\mathcal{L}_{\mathrm{HN}}$};
\node[lblk, right=of hn]       (str)  {Local--global\\similarity consistency};

\draw[arr] (img)--(views); \draw[arr] (views)--(cnn); \draw[arr] (cnn)--(simc);
\draw[arr] (simc)--(mine);  \draw[arr] (mine)--(trip); \draw[arr] (trip)--(vqc);
\draw[arr] (vqc)--(fid);    \draw[arr] (fid)--(inv);
\draw[arr] (mine) |- (marg);\draw[arr] (marg)--(hn);   \draw[arr] (inv)--(marg);
\draw[arr] (simc.east) -- ++(5mm,0) |- (str.east);

\begin{scope}[on background layer]
\node[draw, dashed, gray, rounded corners, fit=(trip)(vqc)(fid),
      label={[gray]above:Quantum similarity space}] {};
\node[draw, dashed, gray, rounded corners, fit=(inv)(marg)(hn)(str),
      label={[gray]below:Condition-aware hard-negative learning}] {};
\end{scope}
\end{tikzpicture}
\caption{CQHN-QDML. Negatives are mined on the classical embedding so the
quantum branch evaluates $O(B)$ pairs rather than $O(B^2)$; the margin widens
with the ambiguity of the mined negative.}
\label{fig:cqhn}
\end{figure}
```

---

## Writing checklist

Before submission, verify each:

- [ ] the dataset section states the crop protocol, instance count and split integrity (F12, F13)
- [ ] the similarity definition appears exactly once and is used consistently (F2)
- [ ] the margin equation matches the prose and matches what was run (F4, D1)
- [ ] the metric dimensionality of every arm is in the results table (D3)
- [ ] the classical baselines are batch-hard, and this is stated
- [ ] every headline number carries `mean ± std` over ≥ 3 seeds
- [ ] the hardware shot budget is reported alongside simulator timings
- [ ] `RE` is captioned as an indicator, never as evidence of advantage
- [ ] no sentence claims quantum advantage unless `advantage_check` licensed it
- [ ] prior art for mixed-state prototypes, perturbed positives and quantum
      triplet loss is cited and credited (§2)
- [ ] the limitations section covers all eight items of spec §15
- [ ] the discussion includes at least one result where the method loses
- [ ] figure 11's two panels use identical projection settings
- [ ] figure 12b includes a failure case
