# Literature: what was supplied, what was found, and where this work sits

Two purposes. First, make the supplied reading usable — each paper reduced to
what it contributes *to this project*, not a generic summary. Second, position
the two algorithms honestly against prior art that the brief does not cite,
because the closest neighbours are the ones a reviewer will find.

PDFs are in `resources/papers/`; extracted text in `resources/extracted_text/`.

---

## 1. Supplied reading

### 1.1 Quantum metric learning — the direct lineage

**Lloyd, Schuld, Ijaz, Izaac, Killoran — *Quantum embeddings for machine
learning*** (arXiv:2001.03622) · `resources/papers/2001.03622v2.pdf`

The origin of the term. Instead of training a measurement on top of a fixed
feature map, train the **embedding itself** to maximally separate classes in
Hilbert space. The payoff is analytic: once the embedding separates the data,
the optimal measurement is known in closed form and depends only on the metric —
the Helström measurement for trace distance, a simple overlap measurement for
Hilbert–Schmidt.

*For this project:* this is the theoretical licence for the whole approach, and
specifically for using **fidelity as the metric** rather than a learned
read-out. It also supplies the argument for F2: the overlap measurement is the
natural pairing with a Hilbert–Schmidt objective, and `(ẑ·ẑ)²` is neither.
Cite in the introduction and again when justifying `sim_mode: fidelity`.

---

**Kim & Bekiranov — *Generalization Performance of Quantum Metric Learning
Classifiers*** (Biomolecules 12(11):1576, 2022) · `resources/papers/biomolecules-12-01576-v4.pdf`

The essential counterweight. Lloyd et al.'s method perfectly separated ants from
bees **on training data**; test performance was never assessed. Kim and
Bekiranov assessed it, on that image data and on breast-cancer clinical data,
and found generalisation holds **only when dimensionality reduction is limited
and the number of model parameters is far below the number of training samples**.

*For this project:* the most operationally useful paper in the set. It is direct
evidence for two design choices here — a very small quantum parameter count
(`L·q·2` = 24 at the default, against ~3,000 training crops) and the caution in
decision **D3** about compressing a 32-D embedding into a 4-D read-out. Cite it
whenever justifying the small circuit, and cite it in the limitations: this is
the failure mode the paper is guarding against.

---

**Haug & Kim — *Generalization of Quantum Machine Learning Models Using Quantum
Fisher Information Metric*** (PRL 133, 050603, 2024) · `resources/papers/PhysRevLett.133.050603.pdf`

Introduces the **data quantum Fisher information metric (DQFIM)**: the rank of
the DQFIM quantifies the circuit depth and the amount of training data needed to
train and generalise, given the ansatz, the data, and their symmetries. Two
counterintuitive results: *breaking* symmetries in the training data can help
generalisation, and out-of-distribution generalisation can beat in-distribution.

*For this project:* the principled way to choose `q` and `L` instead of sweeping
blindly. Two concrete uses — (a) cite it to motivate why the `q × L` sweep
(Table 7) is a capacity study and not just a resource table; (b) the
out-of-distribution result is a direct theoretical hook for the cross-condition
protocol, which is exactly an OOD retrieval setting. If there is time, computing
the DQFIM rank for the chosen ansatz would noticeably strengthen the resource
section.

---

**Zeguendry, Jarir, Quafafou — *Quantum Machine Learning: A Review and Case
Studies*** (Entropy 25(2):287, 2023) · `resources/papers/entropy-25-00287-v2.pdf`

Broad survey aimed at computer scientists, with worked QNN / QSVM / VQC case
studies. Background and encoding-scheme reference; use for the related-work
paragraph on quantum encodings, not for a technical claim.

---

### 1.2 Deep metric learning — the classical half

**Kaya & Bilge — *Deep Metric Learning: A Survey*** (Symmetry 11(9):1066, 2019)
· `resources/papers/symmetry-11-01066-v2.pdf`

Systematic treatment of the three factors that decide DML performance: sampling
strategy, distance metric, and network structure. Siamese and triplet
architectures, with quantitative comparison.

*For this project:* the justification for the PK sampler and batch-hard mining
(F8) — the survey's central message is that **sampling strategy matters as much
as the loss**. Cite when explaining why the baselines are batch-hard rather than
random-triplet, which is what stops the comparison being a strawman.

---

**Mohan, Jawade, Setlur, Govindaraju — *Deep Metric Learning for Computer
Vision: A Brief Overview*** (arXiv:2312.10046) · `resources/papers/2312.10046v1.pdf`

More recent and more focused: objective functions for high intra-class / low
inter-class variance — exactly the substation regime, where two `damaged_cover_plate`
instances can look wildly different and a `fractured_silica_gel_body` can look
much like a `silica_gel_discoloration`.

*For this project:* the cleanest citation for *why metric learning rather than
classification* on this dataset. Its framing of proxy- and prototype-based
losses is also the classical context AQPM's quantum prototypes should be placed
against.

---

**Bellet, Habrard, Sebban — *Metric Learning* (Morgan & Claypool)** ·
`resources/papers/metric-learning-9781627053655-9781627053662_compress.pdf`

The reference monograph — Mahalanobis learning, similarity learning,
generalisation guarantees. Use for formal definitions and for the
generalisation-bound framing in the discussion.

---

### 1.3 Supplied example notebooks

`resources/notebooks/MeL_{1,2,3}.ipynb` — NCA/LMNN on Iris, a triplet network on
synthetic data, and a Siamese contrastive network on Wine. Pedagogical, on toy
tabular data.

*For this project:* useful as a conceptual warm-up and nothing more. Their
pattern — train an embedding, then run kNN in it, and compare against kNN on raw
features — is the structure of the `B0` vs `B1…B6` baseline comparison here, at
a realistic scale.

---

## 2. Work found during this project

None of these are cited in the brief, and the first four are the closest prior
art. Reviewers will find them; the positioning below is what to write.

### 2.1 Direct prior art on quantum metric learning

**QUSL — *Quantum unsupervised image similarity learning with enhanced
performance*** (Yu, Li, Chen, Zhu, Li, Yang; Expert Systems with Applications
258:125112, 2024; [arXiv:2404.02028](https://arxiv.org/abs/2404.02028))

Cited by the brief as `[1]`. Unsupervised similarity learning from triplets,
with positives generated by **perturbing the anchor** — learning independent of
any classical algorithm. Uses metaheuristic search over circuit architectures.
Reports > 50 % reduction in critical quantum resource use and up to 19.5 %
improvement in similarity-detection correlation, with NISQ-hardware validation.

*Positioning.* The nearest neighbour to CQHN-QDML: QUSL's anchor-perturbation
positives are essentially CQHN's dual condition views. **State this explicitly.**
The differences to claim are (i) supervision — QUSL is unsupervised, CQHN uses
class labels for hard-negative mining; (ii) the *adaptive* margin driven by
negative ambiguity; (iii) the explicit condition-invariance objective tied to
measured inspection conditions. Do not claim "first to use perturbed positives
in quantum similarity learning" — QUSL has that.

---

**QAML — *Quantum adversarial metric learning model based on triplet loss
function*** (EPJ Quantum Technology 10, 2023;
[springer](https://link.springer.com/article/10.1140/epjqt/s40507-023-00182-1))

Embeds samples in high-dimensional Hilbert space and optimises a **triplet
loss**, using entanglement and interference to build superposition states of the
triplet so that a *single* parameterised circuit computes sample distances.

*Positioning.* The closest published "quantum triplet loss". Its efficiency
trick — one circuit for a whole triplet — is a genuine alternative to the
approach here and should be acknowledged in related work. AQPM differs in being
prototype-based rather than triplet-based, and in reliability weighting.

---

**Training Parameterized Quantum Circuits with Triplet Loss**
(ECML PKDD 2022; [ACM](https://dl.acm.org/doi/10.1007/978-3-031-26419-1_31))
and **Triplet Loss Based Quantum Encoding for Class Separability**
([arXiv:2509.15705](https://arxiv.org/pdf/2509.15705), Sept 2025)

Triplet-loss training of PQCs, and triplet-driven quantum *encoding* for class
separability. The 2025 paper is the closest thing to a direct precedent for
"train the encoding, not the classifier, with a metric objective".

*Positioning.* Cite both. They establish that metric-objective training of
quantum encodings is an active line — which helps, because it means this project
is contributing to a live conversation rather than claiming an empty field.

---

**Quantum Incremental Learning with Mixed State Prototypes**
([arXiv:2608.10464](https://arxiv.org/pdf/2608.10464))

Uses **mixed-state (density-matrix) class prototypes** in a quantum learning
setting.

*Positioning — important.* This is the precedent for the density-matrix
prototype adopted here under F10. **Cite it and do not claim mixed-state
prototypes as a novelty.** AQPM's contribution is the *reliability weighting* of
the mixture — `ρ_c = Σ r̃_i |ψ_i⟩⟨ψ_i|` with `r̃` learned from the sample — and
the prototype-adaptive margin. Claiming the mixed state itself would be trivially
rebutted.

---

### 2.2 Other brief-cited work

* Yang, Feng & Wei, *Image Similarity Quantum Algorithm and Its Application in
  Image Retrieval Systems*, Entropy 27(2):137, 2025 — brief `[2]`.
* Sha & Binbusayyis, *Effective content-based image retrieval in deep
  reinforcement learning and quantum approach*, Quantum Machine Intelligence 8:75,
  2026 — brief `[3]`.
* *QRLaXAI: quantum representation learning and explainable AI*, Quantum Machine
  Intelligence, 2025 — brief `[5]`. Relevant to the XAI section (§18 of the
  brief); the occlusion-on-a-pair attribution here is a different construction
  and should be contrasted with it.

### 2.3 Application-domain context

For the introduction's "current practice is detection" paragraph:

* *Substation Equipment Defect Detection Based on Improved YOLOv8*, Sensors
  25(11):3410, 2025.
* *YOLOv11-SR: An Enhanced YOLOv11-Based Framework for Defect Detection in
  Complex Substation Inspection Images*.
* *PBZGNet: A Novel Defect Detection Network for Substation Equipment Based on
  Gradual Parallel Branch Architecture*.
* *A Novel Adversarial Deep Learning Method for Substation Defect Image
  Generation*.
* *SubstationAI: Multimodal Large Model-Based Approaches for Analyzing Substation
  Equipment Faults* ([arXiv:2412.17077](https://arxiv.org/pdf/2412.17077)).

*Use:* every one of these is a **detector or classifier**. None produces a
retrieval-capable embedding, none does few-shot or open-set. That gap is the
applied motivation for this paper, and it is worth one sharp paragraph:
inspection teams need "find me other instances that look like this", and a
detector cannot answer it.

---

## 3. Positioning summary — what to claim

| claim | status |
|---|---|
| Quantum metric learning on a real industrial inspection corpus | **defensible** — prior work is on ants/bees, MNIST, Iris, clinical tabular data |
| Reliability-weighted quantum prototypes | **defensible** — mixed-state prototypes exist (arXiv:2608.10464); the learned reliability weighting does not |
| Prototype-adaptive quantum margin driven by inter-prototype confusability | **defensible** |
| Condition-invariant quantum metric tied to *measured* inspection conditions | **defensible** — the derived condition buckets make this testable on real data |
| Classically-guided hard-negative mining to keep quantum cost `O(B)` | **weak** — a natural engineering choice; present as design, not novelty |
| Mixed-state quantum prototypes as such | **do not claim** — arXiv:2608.10464 |
| Perturbed positives for quantum similarity | **do not claim** — QUSL |
| Quantum triplet-style metric loss | **do not claim** — QAML, ECML PKDD 2022 |
| Quantum advantage | **do not claim** — see `advantage_check`, decision D4 |

### The two strongest honest contributions

Neither is in the brief, and both come out of the work in `01_FINDINGS.md`:

1. **A reproducible account of how a hybrid quantum metric model fails
   silently.** F5 (angle-range collapse), F3 (reliability-gate collapse), F7
   (invariance collapse) and F2 (a degenerate similarity that is not a metric)
   are all failures that *train fine and report plausible numbers*. Ablations
   `A9`, `A4b`, `B10` and `A6c` measure each one. Anyone building a hybrid
   quantum metric model will hit at least one of these, and nobody has written
   them down together.

2. **Instance-level quantum retrieval for industrial inspection**, with an
   evaluation protocol that is leakage-checked, condition-bucketed from measured
   image statistics, and paired with an explainability method defined on *pairs*
   rather than on classifier logits — which is what reveals that background
   clutter, not the defect, drives some matches.
