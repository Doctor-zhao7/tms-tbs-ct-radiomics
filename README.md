# TMS–TBS CT radiomics reproducibility code

This repository contains a transparent analysis workflow for the multicentre
study distinguishing *Talaromyces marneffei* spondylitis (TMS) from
tuberculous spondylitis (TBS). It is designed for public release alongside the
manuscript.

## Scope

The code reproduces the **analysis procedure**, not the published numerical
results. Patient-level data and images are not distributed. Exact AUCs,
confidence intervals, and selected feature names can only be reproduced from
the original data.

The implementation fixes the following study decisions:

- outcome: TBS = 1 and TMS = 0;
- cohorts: training, internal validation, and external validation;
- clinical and radiomics Z-score standardisation fitted on training data only;
  within resampling and cross-validation, imputation and scaling are refitted
  using only the corresponding training sample or training fold;
- 1,834 extracted features comprising 360 first-order, 14 shape, and 1,460
  texture features; the extractor validates these counts for every case;
- reproducibility filtering by interobserver ICC(2,1) and intraobserver
  ICC(3,1) at ≥0.80, followed by training-only two-sample t testing, Pearson
  correlation filtering at |r| > 0.90, and ten-fold cross-validated L1
  selection;
- clinical model: MLP with hidden layers (16, 8), ReLU, Adam, alpha = 0.001,
  learning rate = 0.001, batch size = 16, early stopping,
  `validation_fraction = 0.20`, `n_iter_no_change = 50`, and seed 42;
- radiomics and combined models: RBF SVM, C = 10, gamma = 0.01,
  internal sigmoid probability estimation enabled, balanced class weights,
  and seed 42;
- training-derived thresholds fixed at 0.47, 0.52, and 0.49 for the clinical,
  radiomics, and combined models, respectively;
- ROC AUC, DeLong confidence intervals, discrimination metrics, calibration
  plots, decision-curve analysis, paired DeLong comparisons, bootstrap optimism
  correction, repeated nested cross-validation, and sensitivity analyses.

The repeated nested cross-validation uses five outer folds, five inner folds,
and five repeats. For radiomics-containing models, imputation, scaling, t-test
filtering, correlation filtering, and LASSO selection are refitted inside each
training fold. The bootstrap analysis has a narrower scope: it uses
class-stratified resampling and refits the models while holding the final
14-feature set and model settings fixed. It therefore does not represent a
repeat of the complete model-development procedure.

The unified-SVM sensitivity analysis uses the same RBF-SVM family for all
three input sets, with input-specific training-set settings: clinical
`C = 1.0, gamma = scale, threshold = 0.46`; radiomics
`C = 10, gamma = 0.01, threshold = 0.52`; and combined
`C = 10, gamma = 0.01, threshold = 0.49`.

## Upstream basis

The batch-extraction pattern and parameter-file approach were adapted from the
official [PyRadiomics repository](https://github.com/AIM-Harvard/pyradiomics),
revision `8ed579383b44806651c463d5e691f3b2b57522ab` (BSD 3-Clause).

## Input files

`manifest.csv` for extraction:

```text
patient_id,image_path,mask_path
P001,/path/P001_CT.nii.gz,/path/P001_mask.nii.gz
```

The extraction script reorients both CT and segmentation to LPS and verifies
that their size, spacing, origin, and direction match before feature
extraction. Registration and resampling of a mismatched segmentation must be
completed before running the script.

`analysis.csv` for modelling:

```text
patient_id,cohort,outcome,WBC,Hb,CRP,GLB,VAS,spinal_tenderness,original_...,wavelet_...
P001,training,1,8.4,121,32.0,35.2,6,1,...,...
```

Allowed cohort values are `training`, `internal`, and `external`. The six
clinical inputs are WBC, Hb, CRP, GLB, VAS, and spinal tenderness. Radiomics
columns are all remaining numeric columns except identifiers and optional
metadata listed in `config.yaml`.

## Installation and use

The environment is locked to the original analysis versions: Python 3.7.12,
PyRadiomics 3.0.1, and scikit-learn 1.0.2.

```bash
conda env create -f environment.yml
conda activate tms-tbs-radiomics

python scripts/extract_radiomics.py \
  --manifest manifest.csv \
  --params config/pyradiomics_ct.yaml \
  --output radiomics_features.csv

python scripts/calculate_icc.py \
  --reader1-time1 reader1_time1.csv \
  --reader2-time1 reader2_time1.csv \
  --reader1-time2 reader1_time2.csv \
  --output results/icc.csv

python scripts/run_analysis.py \
  --data analysis.csv \
  --icc-results results/icc.csv \
  --config config.yaml \
  --output results

python scripts/run_validation.py \
  --data analysis.csv \
  --icc-results results/icc.csv \
  --config config.yaml \
  --output results \
  --n-jobs -1

python -m unittest discover -s tests
```

`--n-jobs -1` lets the nested grid search use all available CPU cores. Use
`--n-jobs 1` on a shared or memory-constrained system.

## Repository structure

```text
TMS_TBS_reproducible_code/
├── config/
│   └── pyradiomics_ct.yaml
├── examples/
│   ├── analysis_template.csv
│   ├── icc_matrix_template.csv
│   └── manifest_template.csv
├── scripts/
│   ├── calculate_icc.py
│   ├── extract_radiomics.py
│   ├── run_analysis.py
│   └── run_validation.py
├── tests/
│   └── test_core.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── THIRD_PARTY_NOTICES.md
├── config.yaml
├── environment.yml
└── requirements.txt
```

The analysis command creates selected-feature lists, fitted model files,
patient-level predicted probabilities, performance tables, paired DeLong
comparisons, and ROC, calibration, and decision-curve figures. The validation
command creates bootstrap-optimism, repeated nested-cross-validation, and
unified-SVM sensitivity outputs. Never commit patient data, images, direct
identifiers, or local absolute paths.

The strict-reference sensitivity analysis retains culture-positive and/or
molecularly confirmed cases, including targeted PCR/NAAT and mNGS, and applies
the locked models without retraining or threshold re-optimisation.

## Reproducibility safeguards

The scripts check the reported outcome coding, cohort labels, and the full-data
feature-selection sequence of 1,548 ICC-stable features, 1,032 t-test features,
20 correlation-filtered features, and 14 LASSO-selected features. Execution
stops if a full-data count diverges from the expected feature count reported in
the manuscript. Counts are not forced inside resampling folds because valid
training-fold selections can differ from the full-data selection.

## Citation and licence

See `CITATION.cff`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. The repository's
original code is released under BSD 3-Clause.
