# Uncertainty-Label Policies in Chest Radiography — Analysis Code

Code accompanying the manuscript *"Statistical Power and Practical Equivalence of
Uncertainty-Label Resolution in Multi-Label Chest Radiography"*

Everything reported in the paper is reproducible from this code plus the public
source datasets. No image data, model checkpoints or cached embeddings are
redistributed here.

---

## 1. What is and is not included

| Included | Not included |
|---|---|
| All analysis and figure code (`code/`) | CheXpert / NIH images — obtain from the custodians |
| Frozen split manifest with SHA-256 (`splits.json`) | Trained backbone checkpoints |
| Pinned dependency list (`requirements.txt`) | Cached embeddings (~GB; regenerate with `extract.py`) |

The split manifest is the important artifact: it fixes the exact patient-level
partition used in the paper, so a reader with the source data reproduces the same
rows rather than an approximation.

---

## 2. Environment

Python 3.13 with the pinned versions used for the published run:

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All analyses ran on CPU. `tensorflow-metal` was deliberately not installed: its
numerical divergence from the CPU path would invalidate the checkpoint
verification step.

### Dataset location

Project paths are derived from the location of the code, so the repository works
wherever it is cloned. The source datasets live outside the repository and are
expected one level above it:

```
<parent>/
├── Dataset/CheXpert-v1.0-small/
├── Dataset/NIH/
└── <this repository>/code/
```

If your datasets are elsewhere, set `PHD_ROOT` to the directory containing
`Dataset/`, and optionally `PS3_BASE` to the repository root:

```bash
export PHD_ROOT=/path/to/data-parent
```

---

## 3. Layout

`code/` is intentionally flat. The modules import one another directly
(`import heads`), so moving files into subpackages will break them.

**Core**

| File | Role |
|---|---|
| `trunks.py` | Backbone registry, preprocessing spec, embedding paths |
| `preprocessing.py` | Image preprocessing, replicated from the original training scripts |
| `splits.py` | Builds and verifies the frozen patient-level splits |
| `extract.py` | One forward pass per backbone; writes the embedding cache |
| `heads.py` | Policy application, patient resampling, per-pathology head fitting |
| `metrics.py` | Per-class and joint metrics, bootstrap intervals |
| `calibration.py` | Temperature scaling and threshold fitting |
| `multiplicity.py` | Benjamini–Hochberg FDR, shared by every analysis |

**Analysis stages**

| File | Role |
|---|---|
| `tier1.py` | Baseline heads on frozen features |
| `sweep.py` | Policy × pathology × seed sweep |
| `mechanism.py` | Uncertainty burden vs. policy sensitivity |
| `filters.py`, `group_lasso.py`, `controls.py`, `selection_rules.py`, `efficiency.py` | Feature-selection components |
| `phase4_selection.py` | Selection runner (equivalence + efficiency) |
| `conformal.py`, `risk_control.py`, `coverage_shift.py` | Conformal prediction and coverage under shift |
| `phase5_conformal.py` | Conformal runner |
| `external.py`, `fairness.py` | External evaluation and subgroup analysis |
| `phase6_terminal.py` | Terminal evaluation (requires `--confirm-terminal`) |
| `phase6_h3_external.py` | External retention readout for the selection-rule comparison |
| `joint_assembly.py` | Joint metrics and the analytic 3,125-configuration assembly |

**Outputs and reporting**

| File | Role |
|---|---|
| `export_results.py` | Rebuilds every tidy CSV from the result JSONs |
| `export_coverage.py` | Coverage table across backbones |
| `apply_fdr.py` | Applies FDR correction to the exported comparison tables |
| `figures.py`, `figure_power_audit.py` | Publication figures |
| `paper_tables.py` | LaTeX tables |
| `paper1_numbers.py`, `show_joint.py` | Regenerate every number quoted in the paper |

**Verification**

| File | Role |
|---|---|
| `t0c_verify_checkpoint.py` | Confirms the frozen checkpoint reproduces its published performance |
| `t1_verify_aul.py` | Checks a loss implementation against its stated equation |
| `test_joint_assembly.py` | Known-answer tests for the joint-metric assembly |

---

## 4. Reproducing the results

Obtain CheXpert and NIH ChestX-ray14 from their custodians and set the dataset
paths, then:

```bash
cd code

python splits.py --verify                 # confirm the manifest hash matches
python extract.py --trunk densenet121     # ~1.5 h per backbone, CPU
python extract.py --trunk xception

python tier1.py  --trunk xception --C 0.1
python sweep.py  --trunk xception --C 0.1 --eval-on train_c \
                 --tag sweep_train_c_xception_C0.1
python mechanism.py --tag sweep_train_c_xception_C0.1

python phase4_selection.py --trunk xception --C 0.1 --seeds 5 \
       --fractions 0.10,0.25,0.50 --labels all --n-random 5
python phase5_conformal.py --trunk xception --C 0.1 --alpha 0.1
python joint_assembly.py   --trunk xception --C 0.1 --seeds 5

# terminal sets - opened once, deliberately gated
python phase6_terminal.py    --trunk xception --C 0.1 --confirm-terminal
python phase6_h3_external.py --trunk xception --C 0.1 --confirm-terminal

python export_results.py && python apply_fdr.py
python export_coverage.py
python figures.py && python figure_power_audit.py && python paper_tables.py
```

`paper1_numbers.py` then prints every value quoted in the manuscript directly
from the result files.

---

## 5. Protocol constraints enforced in the code

These are not stylistic preferences; relaxing any of them changes the results.

1. Calibration and operating thresholds are fitted on the calibration split only.
2. Regularisation strength is held fixed across policies within a run, so that a
   difference is attributable to the label rule rather than to tuning.
3. Seeds resample patients, not model initialisation — the estimator is convex, so
   re-seeding the optimiser alone would report zero variance.
4. Uncertain labels are preserved end-to-end and transformed only at the point of
   policy application.
5. Rows carrying an uncertain value for the evaluated label are masked, not recoded.
6. Subgroups and pathologies below 30 positive cases are flagged as underpowered
   rather than quoted.
7. Terminal evaluation scripts refuse to run without an explicit confirmation flag
   and record the opening timestamp in their output.

---

## 6. Data availability and licence

CheXpert and NIH ChestX-ray14 remain under their original terms of use and must be
obtained from their custodians. This repository redistributes neither images nor
derived image data.

Code licence: [choose and state, e.g. MIT or Apache-2.0 — add a `LICENSE` file].

## 7. Citation

> [Author list]. Inherited Uncertainty-Label Recommendations in Chest Radiography
> Are Not Supported by the Evidence Used to Derive Them: A Powered, Patient-Disjoint
> Replication. *Diagnostics* [year]. DOI: [to be assigned]

Archived release: [Zenodo DOI once deposited].
