# Dataset: acquisition, conversion, integrity

## 1. The target dataset

**Substation Equipment Appearance Defect Dataset for Complex Inspection Scenarios**

| | |
|---|---|
| DOI | [10.21227/df1t-fz35](https://doi.org/10.21227/df1t-fz35) |
| Host | IEEE DataPort — [dataset page](https://ieee-dataport.org/documents/substation-equipment-appearance-defect-dataset-complex-inspection-scenarios-0) |
| Author | Xuyang Wu |
| Posted | 15 May 2026 |
| Size | 3.59 GB compressed |
| Access | **IEEE DataPort subscription required** — not open access |
| Category | Smart Grid |

**Citation (use verbatim):**

> Xuyang Wu, "Substation Equipment Appearance Defect Dataset for Complex
> Inspection Scenarios", IEEE Dataport, May 15, 2026, doi: 10.21227/df1t-fz35.

```bibtex
@misc{wu2026substation,
  author       = {Wu, Xuyang},
  title        = {Substation Equipment Appearance Defect Dataset for Complex Inspection Scenarios},
  year         = {2026},
  publisher    = {IEEE Dataport},
  doi          = {10.21227/df1t-fz35},
  url          = {https://doi.org/10.21227/df1t-fz35}
}
```

### Published characteristics

| property | value |
|---|---|
| frames | 4,235 |
| categories | 5 |
| resolutions | 200+ distinct, 181×241 → 5184×3888 |
| split | 7:2:1 → 2,965 train / 847 val / 423 test |
| recommended input | 640 × 640 |
| recommended augmentation | mosaic, random flip, colour jitter |
| task as published | object detection (YOLO-style usage) |

### Categories

| idx | canonical name | description |
|---|---|---|
| 0 | `blurred_meter_reading` | gauge dial illegible / out of focus |
| 1 | `damaged_meter_housing` | cracked or chipped instrument housing |
| 2 | `silica_gel_discoloration` | breather desiccant saturated (blue → pink) |
| 3 | `fractured_silica_gel_body` | breather body cracked or broken |
| 4 | `damaged_cover_plate` | electrical box cover bent, missing or damaged |

Index order is **assumed** to match this listing. The converter overrides it
from `data.yaml` / `classes.txt` / `obj.names` if the download ships one —
check `dataset_meta.json → class_names` after the first run and confirm it
against a handful of cropped images before trusting any per-class number.

---

## 2. Getting access

1. Sign in at <https://ieee-dataport.org>. An IEEE account is free; **downloading
   this dataset needs an active DataPort subscription** (IEEE members get it
   bundled with some memberships; institutions often hold one — check your
   university library's IEEE entitlements before paying).
2. Open the [dataset page](https://ieee-dataport.org/documents/substation-equipment-appearance-defect-dataset-complex-inspection-scenarios-0)
   and download the 3.59 GB archive.
3. Extract anywhere; the converter auto-detects the layout.

**If access is blocked or slow**, the options in order of preference:

| option | what it buys | what it costs |
|---|---|---|
| Institutional IEEE entitlement | the real data | an email to your library |
| Email the author (contact via the DataPort page) | the real data; authors often share for academic use | a few days |
| Open power-grid substitutes (below) | real photographs, wrong defect taxonomy | results are about a different dataset |
| `SubstationSim` (built in) | full pipeline validation today | **not a result** |

### Open substitutes, if you need real photographs

| dataset | contents | access |
|---|---|---|
| **InsPLAD** ([arXiv:2311.01619](https://arxiv.org/abs/2311.01619)) | 10,607 UAV power-line images, 17 asset classes, 28,933 instances, 6 defect types with a *classification* subset | public |
| **CPLID** ([GitHub](https://github.com/InsulatorData/InsulatorDataSet)) | Chinese Power Line Insulator Dataset, normal + defective insulators | public |
| **Power-equipment-image-dataset** ([GitHub](https://github.com/xiongsiheng/Power-equipment-image-dataset)) | assorted power-equipment imagery | public |

InsPLAD is the strongest substitute: it is a real inspection corpus with a
defect-classification subset and the same vision challenges (multi-scale,
cluttered background, varied lighting, perspective distortion). If you report on
it, report on it *as InsPLAD* — it is a transfer result, not a result about the
target dataset.

---

## 3. Conversion: detection → metric-learning corpus

```bash
python scripts/01_prepare_data.py --raw "D:/downloads/substation_defect" --out data/processed
```

### What it does, and why

The brief writes the objective over whole images (`I → z`). The dataset is a
detection benchmark: one frame can contain several annotated defects, possibly
of different classes, and the defect often covers under 2 % of a 5184×3888
frame. Asking a 4–8 qubit embedding to represent "a photo that contains a defect
somewhere" is a far weaker learning signal than asking it to represent "a
defect" — and the class label is ill-defined for any multi-class frame.

So each annotated box becomes one **defect instance**:

```
box (x1,y1,x2,y2)
  → expand by `--context 0.15` of the box size on every side
  → square-pad (centred, black) unless `--no-square`
  → resize so the long side ≤ `--save-size` 384 px (never upscales)
  → data/processed/crops/<split>/<class_name>/<12-hex-id>.jpg
```

Boxes whose longer side is below `--min-box 24` px are dropped and counted in
`dataset_meta.json → skipped.tiny_box`. Frames with no annotation are skipped
(they carry no class label).

### Supported input layouts (auto-detected)

| layout | detection rule |
|---|---|
| YOLO | `images/*.jpg` with a matching `labels/*.txt` (`class cx cy w h`, normalised or absolute) |
| Pascal VOC | a matching `*.xml` with `<object><bndbox>` |
| COCO | a `*.json` with `images` / `annotations` / `categories` |
| folder-per-class | `<split>/<class_name>/*.jpg`, whole images used |

### Output

```
data/processed/
├── manifest.csv           one row per crop (see below)
├── dataset_meta.json      layout, split origin, per-class/per-split counts, skip reasons, leakage check
└── crops/<split>/<class_name>/*.jpg
```

`manifest.csv` columns:

| column | meaning |
|---|---|
| `crop_id` | stable hash of `source_image` + box index |
| `path` | path relative to the processed root |
| `label`, `class_name` | class index and canonical name |
| `split` | `train` / `val` / `test` |
| `source_image` | path of the originating frame, relative to the raw root |
| `x1,y1,x2,y2` | expanded box in source-frame pixels |
| `src_w, src_h, src_pixels` | source frame size |
| `crop_w, crop_h` | stored crop size |
| `scale_frac` | box area / frame area — *how small the defect is* |
| `brightness` | mean luminance of the crop, `[0,1]` |
| `contrast` | std of luminance, `[0,1]` |
| `sharpness` | variance of the 3×3 Laplacian — reference-free blur proxy |

---

## 4. Split integrity

Three rules, all enforced and all verifiable:

1. **The published split wins.** If the download carries `train`/`val`/`test`
   directories they are used verbatim. `dataset_meta.json → split_origin` says
   which happened. Never silently re-split a dataset that documents its own
   partition.
2. **No source-frame leakage.** When a split *must* be generated, assignment is
   grouped by source frame and stratified by the frame's majority class. Two
   crops of the same photograph — often the same physical defect under identical
   lighting — can never land on opposite sides of the query/gallery boundary.
3. **Verified, not assumed.** `_leakage_check()` runs at conversion,
   `scripts/02_audit_data.py` re-checks it, additionally hashes every crop to
   find exact duplicates, and flags duplicate groups that span splits. The
   leave-one-out retrieval protocol separately excludes same-frame crops from a
   query's gallery.

Why this is worth three layers: near-duplicate leakage is the single most common
way a retrieval benchmark silently inflates. Recall@1 goes up, the method looks
better, and nothing in the training logs shows it.

---

## 5. Inspection-condition attributes

The dataset carries **no condition labels**, yet the brief asks for
*cross-condition retrieval*. The conversion step therefore measures five
attributes per crop and `condition_buckets()` quantile-buckets any of them into
levels:

| attribute | proxies for |
|---|---|
| `brightness` | exposure / time of day / glare |
| `contrast` | haze, backlighting, low-contrast defects |
| `sharpness` | motion blur, defocus, capture distance |
| `scale_frac` | defect scale — the multi-scale challenge |
| `src_pixels` | the 200+ distinct source resolutions |

`cross_condition_retrieval()` then reports *within-bucket* and *cross-bucket*
retrieval and the gap between them. **That gap is the quantity CQHN-QDML claims
to shrink**, measured on real photographs rather than on synthetic
perturbations — which makes it a stronger result than the perturbation sweep
alone, and it costs nothing extra to compute.

---

## 6. Working before you have access: SubstationSim

```bash
python scripts/01_prepare_data.py --synthetic --sim-images 1200 --out data/processed_sim
```

Renders a structural surrogate: five defect categories with the published
semantics, many distinct resolutions, cluttered backgrounds, non-uniform
illumination lobes, viewing-angle change, defocus, exposure variation, and
one-to-three defect instances per frame at controllable scale.

It writes a **YOLO-layout dataset on disk** and then runs the *same* conversion
the real download will go through, so switching to real data changes exactly one
argument. Every generated corpus carries a `SYNTHETIC.txt` marker.

**What it is for:** making every component runnable, testable and timeable on
day one; serving as the regression fixture for the test suite; letting you tune
the training loop and find the failure modes in `01_FINDINGS.md` before spending
GPU hours on real data.

**What it is not for:** any number in the paper. It is a rendering of what the
dataset is *like*, not a sample of it.

---

## 7. Audit checklist before training

```bash
python scripts/02_audit_data.py --data data/processed --out results/tables
```

Confirm every line:

- [ ] `n_crops` is plausible — expect noticeably more than 4,235 if most frames carry multiple boxes
- [ ] `n_source_frames` ≈ 4,235 (fewer means frames were skipped; check `skipped` in `dataset_meta.json`)
- [ ] all 5 classes present in **all three** splits
- [ ] `split_ratio` ≈ `[0.70, 0.20, 0.10]`; note any deviation in the paper
- [ ] `class_imbalance_ratio` — if > 3:1, report macro-F1 and keep the PK sampler
- [ ] `duplicate_crop_groups` — any group spanning splits must be removed
- [ ] `leakage_check.source_images_in_multiple_splits == 0`
- [ ] `distinct_source_resolutions` > 100 (confirms the heterogeneity the brief relies on)
- [ ] `fraction_defects_under_2pct_of_frame` — quote it; it justifies the crop protocol
- [ ] `class_names` match the canonical list, and spot-check ~10 crops per class by eye

The last item is not optional. If the class index order differs from the
assumed one, every per-class number in the paper is mislabelled and nothing
downstream will tell you.
