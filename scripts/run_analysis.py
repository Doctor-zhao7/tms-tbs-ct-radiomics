#!/usr/bin/env python3
"""Three-cohort analysis for the TMS-versus-TBS study."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, ttest_ind
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score, roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def delong_auc_ci(y: np.ndarray, score: np.ndarray, alpha: float = 0.95):
    """DeLong AUC confidence interval for one binary classifier."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return roc_auc_score(y, score), np.nan, np.nan
    v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
    v01 = np.array([(pos > n).mean() + 0.5 * (pos == n).mean() for n in neg])
    auc = v10.mean()
    var = np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg)
    z = norm.ppf(0.5 + alpha / 2)
    return auc, max(0.0, auc - z * np.sqrt(var)), min(1.0, auc + z * np.sqrt(var))


def delong_components(y, scores):
    """AUC vector and DeLong covariance for one or more paired classifiers."""
    y = np.asarray(y)
    scores = np.atleast_2d(scores)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    v10 = np.empty((scores.shape[0], pos.shape[1]))
    v01 = np.empty((scores.shape[0], neg.shape[1]))
    for k in range(scores.shape[0]):
        comparisons = ((pos[k, :, None] > neg[k, None, :]).astype(float) +
                       0.5 * (pos[k, :, None] == neg[k, None, :]))
        v10[k] = comparisons.mean(axis=1)
        v01[k] = comparisons.mean(axis=0)
    aucs = v10.mean(axis=1)
    covariance = (np.atleast_2d(np.cov(v10, bias=False)) / pos.shape[1] +
                  np.atleast_2d(np.cov(v01, bias=False)) / neg.shape[1])
    return aucs, covariance


def delong_pairwise(y, scores_a, scores_b):
    aucs, covariance = delong_components(y, np.vstack([scores_a, scores_b]))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    z_value = (aucs[0] - aucs[1]) / np.sqrt(max(variance, np.finfo(float).eps))
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z_value)))


def youden_threshold(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def strict_reference_mask(frame):
    """Accept independently recorded positive culture, targeted PCR, or mNGS."""
    evidence = ["culture_positive", "targeted_molecular_positive", "mngs_positive"]
    missing = [column for column in evidence if column not in frame.columns]
    if missing:
        raise ValueError("Strict-reference analysis requires explicit evidence columns: "
                         + ", ".join(missing))
    parsed = []
    for column in evidence:
        values = frame[column].astype(str).str.strip().str.lower()
        if not values.isin({"true", "false", "1", "0"}).all():
            raise ValueError("Evidence column {} must contain only true/false or 1/0; "
                             "unknown is not a negative result".format(column))
        parsed.append(values.isin({"true", "1"}))
    return parsed[0] | parsed[1] | parsed[2]


def manuscript_feature_name(raw_name):
    """Render a PyRadiomics identifier in the notation used by Table S10."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(raw_name)).strip("_")


def audit_s10_names(selected, expected):
    """Compare selected raw identifiers without changing model inputs."""
    normalized = [manuscript_feature_name(name) for name in selected]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Distinct radiomics columns collapse to the same S10 name")
    mapping = dict(zip(normalized, selected))
    rows = [{"s10_feature": name, "selected_raw_feature": mapping.get(name, ""),
             "matched": name in mapping} for name in expected]
    rows.extend({"s10_feature": "", "selected_raw_feature": mapping[name],
                 "matched": False} for name in normalized if name not in expected)
    return pd.DataFrame(rows)


def t_test_select(x: pd.DataFrame, y: pd.Series, p_cutoff: float):
    selected = []
    for col in x.columns:
        _, p = ttest_ind(x.loc[y == 0, col], x.loc[y == 1, col],
                         equal_var=False, nan_policy="omit")
        if np.isfinite(p) and p < p_cutoff:
            selected.append(col)
    return selected


def correlation_filter(x: pd.DataFrame, threshold: float):
    """Deterministic greedy filter in input-column order."""
    corr = x.corr().abs()
    keep = []
    for col in x.columns:
        if all(corr.loc[col, prior] <= threshold for prior in keep):
            keep.append(col)
    return keep


def select_radiomics(x: pd.DataFrame, y: pd.Series, cfg: dict):
    fs = cfg["feature_selection"]
    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x),
                           columns=x.columns, index=x.index)
    stage1 = t_test_select(imputed, y, fs["t_test_p"])
    stage2 = correlation_filter(imputed[stage1], fs["pearson_threshold"])
    if len(stage1) != int(fs["expected_t_test_features"]):
        raise RuntimeError(
            f"The t-test retained {len(stage1)} features; "
            f"the expected feature count reported in the manuscript is "
            f"{fs['expected_t_test_features']}."
        )
    if len(stage2) != int(fs["expected_pearson_features"]):
        raise RuntimeError(
            f"Correlation filtering retained {len(stage2)} features; "
            f"the expected feature count reported in the manuscript is "
            f"{fs['expected_pearson_features']}."
        )
    scaled = StandardScaler().fit_transform(imputed[stage2])
    cv = StratifiedKFold(n_splits=fs["lasso_cv_folds"], shuffle=True,
                        random_state=cfg["random_seed"])
    c_grid = np.logspace(np.log10(fs["lasso_C_min"]),
                         np.log10(fs["lasso_C_max"]),
                         int(fs["lasso_C_values"]))
    l1 = LogisticRegressionCV(Cs=c_grid, cv=cv, penalty="l1",
                              solver="liblinear", scoring=fs["lasso_scoring"],
                              random_state=cfg["random_seed"], max_iter=10000,
                              refit=True)
    l1.fit(scaled, y)
    ranked = pd.Series(np.abs(l1.coef_[0]), index=stage2).sort_values(ascending=False)
    nonzero = ranked[ranked > 0]
    expected = int(fs["expected_features"])
    if len(nonzero) != expected:
        raise RuntimeError(
            f"L1 retained {len(nonzero)} features, expected {expected}. "
            "The configured feature-selection sequence did not reproduce "
            "the expected feature count reported in the manuscript."
        )
    audit = {"t_test": len(stage1), "pearson": len(stage2),
             "lasso": len(nonzero), "selected_C": float(l1.C_[0])}
    return list(nonzero.index), audit


def clinical_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                              solver="adam", alpha=0.001,
                              learning_rate_init=0.001, batch_size=16,
                              max_iter=1000, early_stopping=True,
                              validation_fraction=0.20,
                              n_iter_no_change=50,
                              random_state=seed)),
    ])


def svm_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.01, probability=True,
                    class_weight="balanced", random_state=seed)),
    ])


def combined_model(seed, clinical_columns, radiomics_columns):
    """Standardise clinical and radiomics inputs using training data."""
    preprocessing = ColumnTransformer([
        ("clinical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), clinical_columns),
        ("radiomics", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), radiomics_columns),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.01, probability=True,
                    class_weight="balanced", random_state=seed)),
    ])


def metrics_row(model_name, cohort, y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc, lo, hi = delong_auc_ci(np.asarray(y), np.asarray(score))
    return {"model": model_name, "cohort": cohort, "n": len(y),
            "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
            "threshold": threshold, "accuracy": accuracy_score(y, pred),
            "tbs_sensitivity": tp / (tp + fn), "tbs_specificity": tn / (tn + fp),
            "ppv": precision_score(y, pred, zero_division=0),
            "npv": tn / (tn + fn) if (tn + fn) else 0.0,
            "f1": f1_score(y, pred, zero_division=0),
            "tms_sensitivity": tn / (tn + fp),
            "tms_specificity": tp / (tp + fn), "tn": tn, "fp": fp,
            "fn": fn, "tp": tp}


def save_plots(predictions, out):
    for plot_type in ("roc", "calibration", "dca"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, cohort in zip(axes, ["training", "internal", "external"]):
            part = predictions[predictions.cohort == cohort]
            for model in ["clinical", "radiomics", "combined"]:
                y, p = part.outcome.to_numpy(), part[model].to_numpy()
                if plot_type == "roc":
                    fpr, tpr, _ = roc_curve(y, p)
                    ax.plot(fpr, tpr, label=f"{model} ({roc_auc_score(y,p):.3f})")
                    ax.plot([0, 1], [0, 1], "k--", lw=.8)
                    ax.set(xlabel="1 - specificity", ylabel="sensitivity")
                elif plot_type == "calibration":
                    frac, mean = calibration_curve(y, p, n_bins=6, strategy="quantile")
                    ax.plot(mean, frac, marker="o", label=model)
                    ax.plot([0, 1], [0, 1], "k--", lw=.8)
                    ax.set(xlabel="predicted probability", ylabel="observed frequency")
                else:
                    thresholds = np.linspace(.01, .99, 99)
                    n = len(y)
                    net = [((p >= t) & (y == 1)).sum()/n -
                           ((p >= t) & (y == 0)).sum()/n * t/(1-t)
                           for t in thresholds]
                    ax.plot(thresholds, net, label=model)
                    ax.set(xlabel="threshold probability", ylabel="net benefit")
            if plot_type == "dca":
                prevalence = y.mean()
                treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
                ax.plot(thresholds, treat_all, "k--", lw=1, label="treat all")
                ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1,
                        label="treat none")
            ax.set_title(cohort.capitalize())
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{plot_type}.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--icc-results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = pd.read_csv(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    outcome, cohort = cfg["outcome_column"], cfg["cohort_column"]
    if not set(df[outcome].dropna().unique()).issubset({0, 1}):
        raise ValueError("Outcome must be coded TMS=0 and TBS=1")
    train = df[df[cohort] == "training"].copy()
    clinical = cfg["clinical_features"]
    excluded = {cfg["id_column"], outcome, cohort, *clinical,
                *cfg.get("metadata_columns", [])}
    radiomics = [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]
    icc_results = pd.read_csv(args.icc_results)
    required_icc_columns = {"feature", "interobserver_icc_2_1",
                            "intraobserver_icc_3_1", "passed"}
    if not required_icc_columns.issubset(icc_results.columns):
        raise ValueError(f"ICC file must contain {sorted(required_icc_columns)}")
    passed = icc_results.loc[
        icc_results["passed"].astype(str).str.lower().eq("true"), "feature"]
    radiomics = [column for column in radiomics if column in set(passed)]
    expected_icc = int(cfg["feature_selection"]["expected_icc_features"])
    if len(radiomics) != expected_icc:
        raise RuntimeError(
            f"ICC filtering retained {len(radiomics)} available features; "
            f"the expected feature count reported in the manuscript is "
            f"{expected_icc}."
        )
    selected, audit = select_radiomics(train[radiomics], train[outcome], cfg)
    audit = {"icc": len(radiomics), **audit}
    (args.output / "selected_radiomics_features.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8")
    expected_names = cfg.get("reported_s10_features", [])
    if expected_names:
        comparison = audit_s10_names(selected, expected_names)
        comparison.to_csv(args.output / "s10_feature_name_audit.csv", index=False)
        audit["s10_names_match"] = bool(comparison["matched"].all())
    (args.output / "selection_audit.json").write_text(json.dumps(audit, indent=2))

    models = {"clinical": (clinical_model(cfg["random_seed"]), clinical),
              "radiomics": (svm_model(cfg["random_seed"]), selected),
              "combined": (combined_model(cfg["random_seed"], clinical, selected),
                           clinical + selected)}
    thresholds, threshold_audit, all_predictions, rows = {}, [], [], []
    for name, (model, cols) in models.items():
        model.fit(train[cols], train[outcome])
        train_score = model.predict_proba(train[cols])[:, 1]
        configured = cfg["models"].get(f"{name}_threshold")
        calculated = youden_threshold(train[outcome], train_score)
        thresholds[name] = float(configured) if configured is not None else calculated
        threshold_audit.append({"model": name, "training_youden_threshold": calculated,
                                "applied_threshold": thresholds[name],
                                "source": "locked_config" if configured is not None else "training_youden",
                                "difference": thresholds[name] - calculated})
        joblib.dump({"model": model, "columns": cols, "threshold": thresholds[name]},
                    args.output / f"{name}_model.joblib")

    pd.DataFrame(threshold_audit).to_csv(args.output / "threshold_audit.csv", index=False)

    for cohort_name in ["training", "internal", "external"]:
        part = df[df[cohort] == cohort_name].copy()
        keep_columns = [cfg["id_column"], outcome]
        keep_columns += [column for column in
                         ("reference_standard", "culture_positive",
                          "targeted_molecular_positive", "mngs_positive")
                         if column in part.columns]
        pred = part[keep_columns].rename(columns={outcome: "outcome"})
        pred["cohort"] = cohort_name
        for name, (model, cols) in models.items():
            score = model.predict_proba(part[cols])[:, 1]
            pred[name] = score
            rows.append(metrics_row(name, cohort_name, part[outcome], score, thresholds[name]))
        all_predictions.append(pred)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output / "performance.csv", index=False)
    pairwise = []
    for cohort_name in ["training", "internal", "external"]:
        part = predictions[predictions.cohort == cohort_name]
        for first, second in [("radiomics", "clinical"),
                              ("combined", "clinical"),
                              ("combined", "radiomics")]:
            auc1, auc2, p_value = delong_pairwise(
                part.outcome.to_numpy(), part[first].to_numpy(),
                part[second].to_numpy())
            pairwise.append({"cohort": cohort_name, "model_1": first,
                             "model_2": second, "auc_1": auc1,
                             "auc_2": auc2, "p_value": p_value})
    pd.DataFrame(pairwise).to_csv(args.output / "delong_pairwise.csv", index=False)

    evidence = ("culture_positive", "targeted_molecular_positive", "mngs_positive")
    if any(column in df.columns for column in evidence):
        eligible_reference = strict_reference_mask(predictions)
        strict_rows = []
        cohort_counts = []
        for cohort_name in ["training", "internal", "external"]:
            mask = ((predictions.cohort == cohort_name) &
                    eligible_reference)
            part = predictions[mask]
            cohort_counts.append({"cohort": cohort_name,
                                  "included": int(mask.sum()),
                                  "excluded": int((predictions.cohort == cohort_name).sum() - mask.sum())})
            if part.empty or part.outcome.nunique() < 2:
                continue
            for name in ["clinical", "radiomics", "combined"]:
                strict_rows.append(metrics_row(name, cohort_name, part.outcome,
                                               part[name], thresholds[name]))
        pd.DataFrame(strict_rows).to_csv(
            args.output / "strict_reference_performance.csv", index=False)
        pd.DataFrame(cohort_counts).to_csv(
            args.output / "strict_reference_counts.csv", index=False)
    else:
        raise ValueError("Strict-reference analysis requires culture_positive, "
                         "targeted_molecular_positive and mngs_positive; "
                         "free-text reference_standard cannot establish positivity")
    save_plots(predictions, args.output)


if __name__ == "__main__":
    main()
