# Early Sepsis Prediction from ICU Time-Series Data

A comparative evaluation of gradient boosting and deep sequence models for predicting sepsis onset in advance of clinical recognition, using the PhysioNet/Computing in Cardiology 2019 Challenge dataset.

---

## Overview

Sepsis accounts for an estimated 11 million deaths annually — roughly 19.7% of all global deaths. Delayed antimicrobial treatment is associated with increased mortality, which motivates automated early-warning systems that flag deterioration before clinical recognition.

This repository implements and compares six modelling approaches on routinely collected ICU data:

| Model | Family |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Bagged trees |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| Transformer encoder | Attention |
| BiLSTM-MHA | Recurrent + multi-head attention |

Beyond discrimination metrics, the pipeline includes SHAP and integrated-gradients attribution, Monte Carlo dropout uncertainty quantification, decision-curve analysis, multi-horizon lead-time analysis, and temporal drift validation.

> **Status:** research code, actively under revision. Please read [Known Limitations](#known-limitations) before interpreting or reusing any reported metric.

---

## Dataset

Data come from the **PhysioNet/CinC 2019 Challenge**, Training Set A:

- **Source:** Beth Israel Deaconess Medical Center
- **Patients:** 20,336 ICU admissions
- **Format:** one pipe-delimited `.psv` file per patient, one row per hour
- **Variables:** 8 vital signs, 26 laboratory values, 6 demographic/administrative fields
- **Label:** `SepsisLabel`, pre-shifted six hours ahead of Sepsis-3 onset

The data are publicly available and require no credentialing:

```bash
wget -O training_setA.zip \
  https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip
unzip training_setA.zip
```

On Kaggle, attach the dataset through **+ Add Input → Datasets** instead; no internet access is needed.

### Provenance verification

If you obtain the data via a mirror rather than PhysioNet directly, verify it before use:

```python
# 1. File count      — Set A should contain exactly 20,336 .psv files
# 2. Naming          — Set A files are p00xxxx.psv; Set B files are p1xxxxx.psv
# 3. Schema          — 41 columns in the documented order
# 4. Sparsity        — labs must remain sparse: Lactate ~3%, Fibrinogen ~1% of hours.
#                      Values near 100% indicate a pre-imputed mirror, which would
#                      invalidate any analysis of measurement missingness.
```

Cite PhysioNet, never the mirror.

---

## Repository structure

```
.
├── notebooks/
│   └── sepsis_prediction.ipynb      # end-to-end pipeline (17 cells)
├── results/
│   ├── FINAL_all_metrics.csv        # model comparison
│   ├── FINAL_results_table.csv      # extended metrics
│   └── cv_5fold_smote_results.csv   # cross-validation
├── figures/
│   ├── fig1_model_comparison.png    # ROC, PR, calibration
│   ├── fig3_shap_*.png              # feature attribution
│   ├── fig4_lead_time_analysis.png  # 3/6/12 h horizons
│   ├── fig5_decision_curve.png      # net benefit
│   ├── fig6_uncertainty_analysis.png
│   └── fig7_temporal_validation.png
├── models/
│   ├── best_bilstm_v2.pt
│   └── best_transformer.pt
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, scikit-learn 1.3+, XGBoost 2.0+, LightGBM 4.0+, imbalanced-learn, SHAP, pandas, numpy, matplotlib.

A CUDA-capable GPU is recommended. The deep models train in roughly 10–20 minutes on a single T4; CPU-only training is substantially slower.

---

## Methodology

### Preprocessing

1. **Forward-fill within patient.** Carries the last observed value forward, mirroring how a clinician reads a chart. Backward-filling is never applied, as it would leak future information.
2. **Median imputation** for values never observed for a patient, with medians computed on the training fold only.
3. **Feature engineering** — the raw variables are augmented with:
   - first-difference terms capturing rate of change
   - rolling-window statistics over recent hours
   - shock index (HR / SBP)
   - a SOFA proxy score

### Sequence construction

Multi-hour sliding windows are generated per patient, each labelled by the outcome at its final timestep.

### Class imbalance

Positive windows represent a small minority of the dataset. SMOTE is applied to training data, and class weighting is used for the neural models.

### Evaluation

**AUPRC is the primary metric.** At low positive prevalence, AUROC is optimistically biased, and the meaningful AUPRC baseline is the prevalence itself rather than 0.5. Reported alongside: AUROC, MCC, precision, recall, F1, Brier score, calibration curves, and decision-curve net benefit.

---

## Known limitations

### 1. Data partitioning (must be addressed before results are cited)

**The current pipeline partitions at the window level rather than the patient level.** Because sliding windows overlap heavily, windows from the same patient appear in training, validation and test sets simultaneously. High-capacity models can therefore recognise individual patients rather than learn generalisable physiological patterns.

Three observations from this repository's own outputs indicate the effect is material:

| Evidence | Observation |
|---|---|
| Same model, different evaluations | XGBoost scores AUROC 0.9333 (window-level split), 0.8524 (5-fold CV), and 0.755 (lead-time analysis) |
| Ranking tracks capacity to memorise | Logistic regression, which cannot memorise a patient, reaches AUPRC 0.050; restricting XGBoost to 10 features drops AUPRC from 0.334 to 0.080 |
| Decision-curve analysis | XGBoost and LightGBM show negative net benefit at every threshold, which is inconsistent with genuine AUPRC of 0.33 at ~2% prevalence |

For context, a 2025 systematic review of 91 sepsis prediction models reported a pooled full-window internal median AUROC of 0.811 (IQR 0.760–0.842), falling to 0.783 externally, with official Utility Scores dropping from 0.381 internally to −0.164 externally.

**Fix.** Partition the unique patient list and assign every window from a patient to exactly one fold:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Patient-level outcome for stratification
uniq_pids = np.unique(pids_seq)
pid_label = np.array([y_seq[pids_seq == p].max() for p in uniq_pids])

pid_tr, pid_tmp = train_test_split(
    uniq_pids, test_size=0.30, random_state=42, stratify=pid_label)
tmp_label = np.array([y_seq[pids_seq == p].max() for p in pid_tmp])
pid_va, pid_te = train_test_split(
    pid_tmp, test_size=0.50, random_state=42, stratify=tmp_label)

m_tr, m_va, m_te = (np.isin(pids_seq, pid_tr),
                    np.isin(pids_seq, pid_va),
                    np.isin(pids_seq, pid_te))

X_tr, y_tr = X_seq[m_tr], y_seq[m_tr]
X_va, y_va = X_seq[m_va], y_seq[m_va]
X_te, y_te = X_seq[m_te], y_seq[m_te]

# Prove the folds are disjoint
assert not (set(pid_tr) & set(pid_va))
assert not (set(pid_tr) & set(pid_te))
assert not (set(pid_va) & set(pid_te))
```

Expect AUROC in the approximate range 0.75–0.85 after correction. That reduction reflects the removal of leakage, not a regression in model quality.

### 2. SMOTE placement

Confirm that SMOTE is fitted inside training folds only. Applying it before partitioning propagates synthetic samples derived from test observations into the training set.

### 3. Feature attribution reflects care process as well as physiology

SHAP ranks administrative variables — time from hospital admission to ICU admission, and ICU unit type — above established physiological markers such as lactate and leukocyte count. This is partly expected: the Sepsis-3 label depends on antibiotic and blood-culture ordering times, so it encodes clinician behaviour alongside pathophysiology. It should be reported rather than treated as an artefact.

### 4. Additional constraints

- **Single-site design.** Only Set A (Beth Israel Deaconess) is used. External validation on Set B (Emory University Hospital) has not been performed.
- **Incomplete SOFA.** The dataset lacks PaO₂, Glasgow Coma Scale, urine output and vasopressor dosing, so full SOFA cannot be computed and a proxy is used.
- **Single random seed.** Results are not yet averaged across seeds. Differences of roughly 0.01 between models are within noise.
- **Not comparable to the 2019 leaderboard.** The Challenge's sequestered third hospital system is not publicly available.

---

## Roadmap

- [ ] Patient-level partitioning with disjointness assertions
- [ ] Re-run all models and regenerate figures
- [ ] Five-seed repetition reporting mean ± SD with bootstrap confidence intervals
- [ ] Official PhysioNet Utility Score alongside AUPRC
- [ ] External validation on Training Set B
- [ ] Completed TRIPOD+AI reporting checklist

---

## Citation

If you use this code, please cite the underlying dataset:

```bibtex
@article{reyna2020sepsis,
  title   = {Early Prediction of Sepsis From Clinical Data:
             The {PhysioNet/Computing} in Cardiology Challenge 2019},
  author  = {Reyna, Matthew A. and Josef, Christopher S. and Jeter, Russell
             and Shashikumar, Supreeth P. and Westover, M. Brandon
             and Nemati, Shamim and Clifford, Gari D. and Sharma, Ashish},
  journal = {Critical Care Medicine},
  volume  = {48},
  number  = {2},
  pages   = {210--217},
  year    = {2020},
  doi     = {10.1097/CCM.0000000000004145}
}
```

PhysioNet resource DOI: [`10.13026/v64v-d857`](https://doi.org/10.13026/v64v-d857)

---

## Key references

1. Rudd KE, Johnson SC, Agesa KM, et al. Global, regional, and national sepsis incidence and mortality, 1990–2017. *Lancet.* 2020;395(10219):200–211. [doi:10.1016/S0140-6736(19)32989-7](https://doi.org/10.1016/S0140-6736(19)32989-7)
2. Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. *npj Digit Med.* 2025;8:190. [doi:10.1038/s41746-025-01587-1](https://doi.org/10.1038/s41746-025-01587-1)
3. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Intern Med.* 2021;181(8):1065–1070. [doi:10.1001/jamainternmed.2021.2626](https://doi.org/10.1001/jamainternmed.2021.2626)
4. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA.* 2016;315(8):801–810. [doi:10.1001/jama.2016.0287](https://doi.org/10.1001/jama.2016.0287)
5. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ.* 2024;385:e078378. [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
6. Brookshire G, Kasper J, Blauch NM, et al. Data leakage in deep learning studies of translational EEG. *Front Neurosci.* 2024;18:1373515. [doi:10.3389/fnins.2024.1373515](https://doi.org/10.3389/fnins.2024.1373515)
7. Fleuren LM, Klausch TLT, Zwager CL, et al. Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. *Intensive Care Med.* 2020;46(3):383–400. [doi:10.1007/s00134-019-05872-y](https://doi.org/10.1007/s00134-019-05872-y)
8. Tang F, Yuan H, Li X, Qiao L. Effect of delayed antibiotic use on mortality outcomes in patients with sepsis or septic shock. *Int Immunopharmacol.* 2024;129:111616. [doi:10.1016/j.intimp.2024.111616](https://doi.org/10.1016/j.intimp.2024.111616)

---

## Authors

**Diana Opiyo** — Department of Information Sciences and Technologies, College of Computing, Grand Valley State University, Allendale, Michigan, USA

**Edna Ovulley** — Southern Arkansas University, Magnolia, Arkansas, USA

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

The PhysioNet/CinC 2019 Challenge data carry their own terms; consult the [PhysioNet project page](https://physionet.org/content/challenge-2019/1.0.0/) before redistribution.

---

## Disclaimer

Research code. Not a medical device, and not validated for clinical use. No output from this repository should inform patient care.
