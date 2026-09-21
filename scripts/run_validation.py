#!/usr/bin/env python3
"""Bootstrap, leakage-safe nested-CV, and SVM sensitivity analyses."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_analysis import (clinical_model, combined_model, correlation_filter,
                          metrics_row, svm_model)


class FoldRadiomicsSelector(BaseEstimator, TransformerMixin):
    """Fit all radiomics preprocessing and selection within a CV fold."""

    def __init__(self, feature_columns, selection_config, seed=42):
        self.feature_columns = feature_columns
        self.selection_config = selection_config
        self.seed = seed

    def fit(self, x, y):
        columns = list(self.feature_columns)
        frame = x[columns]
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = pd.DataFrame(self.imputer_.fit_transform(frame),
                               columns=columns, index=frame.index)
        y_array = np.asarray(y)
        p_values = {}
        for column in columns:
            _, p_value = ttest_ind(imputed.loc[y_array == 0, column],
                                   imputed.loc[y_array == 1, column],
                                   equal_var=False, nan_policy="omit")
            p_values[column] = p_value
        stage1 = [column for column in columns
                  if np.isfinite(p_values[column]) and
                  p_values[column] < self.selection_config["t_test_p"]]
        if not stage1:
            finite = {key: value for key, value in p_values.items()
                      if np.isfinite(value)}
            if not finite:
                raise RuntimeError("No radiomics feature can be tested in this fold")
            stage1 = [min(finite, key=finite.get)]
        stage2 = correlation_filter(
            imputed[stage1], self.selection_config["pearson_threshold"])
        scaled = StandardScaler().fit_transform(imputed[stage2])
        class_counts = pd.Series(y_array).value_counts()
        cv_folds = min(int(self.selection_config["lasso_cv_folds"]),
                       int(class_counts.min()))
        if cv_folds < 2:
            raise RuntimeError("At least two observations per class are required")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=self.seed)
        c_grid = np.logspace(
            np.log10(self.selection_config["lasso_C_min"]),
            np.log10(self.selection_config["lasso_C_max"]),
            int(self.selection_config["lasso_C_values"]))
        selector = LogisticRegressionCV(
            Cs=c_grid, cv=cv, penalty="l1", solver="liblinear",
            scoring=self.selection_config["lasso_scoring"],
            random_state=self.seed, max_iter=10000, refit=True)
        selector.fit(scaled, y_array)
        coefficients = pd.Series(np.abs(selector.coef_[0]), index=stage2)
        selected = list(coefficients[coefficients > 0].index)
        self.selected_features_ = selected or [coefficients.idxmax()]
        self.output_scaler_ = StandardScaler().fit(
            imputed[self.selected_features_])
        return self

    def transform(self, x):
        frame = x[list(self.feature_columns)]
        imputed = pd.DataFrame(self.imputer_.transform(frame),
                               columns=list(self.feature_columns),
                               index=frame.index)
        return self.output_scaler_.transform(imputed[self.selected_features_])


class FoldCombinedTransformer(BaseEstimator, TransformerMixin):
    """Scale clinical data and select/scale radiomics entirely within-fold."""

    def __init__(self, clinical_columns, radiomics_columns,
                 selection_config, seed=42):
        self.clinical_columns = clinical_columns
        self.radiomics_columns = radiomics_columns
        self.selection_config = selection_config
        self.seed = seed

    def fit(self, x, y):
        self.clinical_imputer_ = SimpleImputer(strategy="median")
        clinical = self.clinical_imputer_.fit_transform(
            x[list(self.clinical_columns)])
        self.clinical_scaler_ = StandardScaler().fit(clinical)
        self.radiomics_selector_ = FoldRadiomicsSelector(
            self.radiomics_columns, self.selection_config, self.seed)
        self.radiomics_selector_.fit(x, y)
        return self

    def transform(self, x):
        clinical = self.clinical_scaler_.transform(
            self.clinical_imputer_.transform(x[list(self.clinical_columns)]))
        radiomics = self.radiomics_selector_.transform(x)
        return np.column_stack([clinical, radiomics])


def make_nested_candidate(name, seed, input_name, clinical, radiomics, fs):
    if input_name == "combined":
        preprocessing = FoldCombinedTransformer(clinical, radiomics, fs, seed)
    elif input_name == "radiomics":
        preprocessing = FoldRadiomicsSelector(radiomics, fs, seed)
    else:
        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    if name == "lr":
        estimator = LogisticRegression(max_iter=10000, random_state=seed)
        grid = {"model__C": [0.01, 0.1, 1, 10, 100]}
    elif name == "svm":
        estimator = SVC(probability=True, random_state=seed)
        grid = {"model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.001, 0.01, 0.1],
                "model__kernel": ["rbf"],
                "model__class_weight": [None, "balanced"]}
    elif name == "mlp":
        estimator = MLPClassifier(
            max_iter=1000, early_stopping=True, validation_fraction=0.20,
            n_iter_no_change=50, random_state=seed)
        grid = {"model__hidden_layer_sizes": [(8,), (16,), (8, 4), (16, 8)],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__learning_rate_init": [0.0005, 0.001, 0.005],
                "model__batch_size": [16, 32]}
    else:
        raise ValueError(name)
    return Pipeline([("preprocessing", preprocessing),
                     ("model", estimator)]), grid


def bootstrap_optimism(x, y, factory, iterations, seed):
    """Estimate optimism by class-stratified resampling of a fixed model."""
    rng = np.random.RandomState(seed)
    apparent_model = factory()
    apparent_model.fit(x, y)
    apparent_auc = roc_auc_score(y, apparent_model.predict_proba(x)[:, 1])
    optimism = []
    y_array = np.asarray(y)
    class_indices = [np.flatnonzero(y_array == value)
                     for value in np.unique(y_array)]
    for _ in range(iterations):
        index = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        rng.shuffle(index)
        model = factory()
        model.fit(x.iloc[index], y.iloc[index])
        boot_auc = roc_auc_score(y.iloc[index],
                                 model.predict_proba(x.iloc[index])[:, 1])
        original_auc = roc_auc_score(y, model.predict_proba(x)[:, 1])
        optimism.append(boot_auc - original_auc)
    mean_optimism = float(np.mean(optimism))
    return apparent_auc, mean_optimism, apparent_auc - mean_optimism


def fixed_svm(input_name, seed, clinical, radiomics, parameters):
    if input_name == "clinical":
        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    elif input_name == "radiomics":
        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        preprocessing = ColumnTransformer([
            ("clinical", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), clinical),
            ("radiomics", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), radiomics),
        ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", SVC(kernel="rbf", C=float(parameters["C"]),
                      gamma=parameters["gamma"], probability=True,
                      class_weight="balanced", random_state=seed)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--icc-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel jobs for the inner GridSearchCV (-1 uses all CPUs)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = pd.read_csv(args.data)
    train = df[df[cfg["cohort_column"]] == "training"].reset_index(drop=True)
    y = train[cfg["outcome_column"]]
    clinical = cfg["clinical_features"]
    selected_path = args.output / "selected_radiomics_features.txt"
    selected = [line.strip() for line in selected_path.read_text().splitlines()
                if line.strip()]
    icc_results = pd.read_csv(args.icc_results)
    passed = set(icc_results.loc[
        icc_results["passed"].astype(str).str.lower().eq("true"), "feature"])
    excluded = {cfg["id_column"], cfg["outcome_column"],
                cfg["cohort_column"], *clinical,
                *cfg.get("metadata_columns", [])}
    all_radiomics = [column for column in
                      df.select_dtypes(include=np.number).columns
                      if column not in excluded and column in passed]
    expected_icc = int(cfg["feature_selection"]["expected_icc_features"])
    if len(all_radiomics) != expected_icc:
        raise RuntimeError(
            "Nested CV must begin with all ICC-passed radiomics features; "
            "found {}, expected {} as reported in the manuscript.".format(
                len(all_radiomics), expected_icc))
    seed = cfg["random_seed"]
    iterations = int(cfg["validation"]["bootstrap_iterations"])
    args.output.mkdir(parents=True, exist_ok=True)

    definitions = {
        "clinical": (clinical, lambda: clinical_model(seed)),
        "radiomics": (selected, lambda: svm_model(seed)),
        "combined": (clinical + selected,
                     lambda: combined_model(seed, clinical, selected)),
    }
    bootstrap_rows = []
    for name, (columns, factory) in definitions.items():
        apparent, optimism, corrected = bootstrap_optimism(
            train[columns], y, factory, iterations, seed)
        bootstrap_rows.append({
            "model": name, "apparent_auc": apparent,
            "mean_optimism": optimism,
            "optimism_corrected_auc": corrected,
            "bootstrap_iterations": iterations,
            "resampling": "class-stratified",
            "model_scope": "fixed final features and settings; model refitted",
        })
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output / "bootstrap_optimism.csv", index=False)

    outer = RepeatedStratifiedKFold(
        n_splits=cfg["validation"]["nested_cv_outer_folds"],
        n_repeats=cfg["validation"]["nested_cv_repeats"], random_state=seed)
    inner = StratifiedKFold(
        n_splits=cfg["validation"]["nested_cv_inner_folds"],
        shuffle=True, random_state=seed)
    nested_rows = []
    for input_name, columns in [
            ("clinical", clinical),
            ("radiomics", all_radiomics),
            ("combined", clinical + all_radiomics)]:
        x = train[columns]
        for algorithm in ["lr", "svm", "mlp"]:
            fold_scores = []
            for train_index, test_index in outer.split(x, y):
                estimator, grid = make_nested_candidate(
                    algorithm, seed, input_name, clinical, all_radiomics,
                    cfg["feature_selection"])
                search = GridSearchCV(estimator, grid, scoring="roc_auc",
                                      cv=inner, n_jobs=args.n_jobs, refit=True)
                search.fit(x.iloc[train_index], y.iloc[train_index])
                score = search.predict_proba(x.iloc[test_index])[:, 1]
                fold_scores.append(roc_auc_score(y.iloc[test_index], score))
            nested_rows.append({
                "input_set": input_name, "algorithm": algorithm,
                "mean_auc": float(np.mean(fold_scores)),
                "sd_auc": float(np.std(fold_scores, ddof=1)),
                "outer_evaluations": len(fold_scores),
                "feature_selection_scope": "refitted within every fold",
            })
    pd.DataFrame(nested_rows).to_csv(
        args.output / "repeated_nested_cv.csv", index=False)

    unified_rows = []
    svm_parameters = cfg["models"]["unified_svm"]
    for input_name, columns in [
            ("clinical", clinical),
            ("radiomics", selected),
            ("combined", clinical + selected)]:
        estimator = fixed_svm(input_name, seed, clinical, selected,
                              svm_parameters[input_name])
        estimator.fit(train[columns], y)
        for cohort_name in ["training", "internal", "external"]:
            part = df[df[cfg["cohort_column"]] == cohort_name]
            score = estimator.predict_proba(part[columns])[:, 1]
            threshold = float(svm_parameters[input_name]["threshold"])
            row = metrics_row(input_name, cohort_name,
                              part[cfg["outcome_column"]], score, threshold)
            row["input_set"] = row.pop("model")
            row["C"] = svm_parameters[input_name]["C"]
            row["gamma"] = svm_parameters[input_name]["gamma"]
            unified_rows.append(row)
    pd.DataFrame(unified_rows).to_csv(
        args.output / "unified_svm_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
