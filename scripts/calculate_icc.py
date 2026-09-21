#!/usr/bin/env python3
"""Calculate interobserver ICC(2,1) and intraobserver ICC(3,1)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def icc(values: np.ndarray, kind: str) -> float:
    """Shrout-Fleiss single-measure ICC for a complete n-by-k matrix."""
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    if kind == "icc2":
        denominator = (ms_rows + (k - 1) * ms_error +
                       k * (ms_cols - ms_error) / n)
    elif kind == "icc3":
        denominator = ms_rows + (k - 1) * ms_error
    else:
        raise ValueError("kind must be icc2 or icc3")
    return float((ms_rows - ms_error) / denominator)


def read_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "patient_id" not in data.columns:
        raise ValueError(f"{path} must contain patient_id")
    return data.set_index("patient_id").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader1-time1", required=True, type=Path)
    parser.add_argument("--reader2-time1", required=True, type=Path)
    parser.add_argument("--reader1-time2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    r1t1 = read_matrix(args.reader1_time1)
    r2t1 = read_matrix(args.reader2_time1)
    r1t2 = read_matrix(args.reader1_time2)
    common_subjects = r1t1.index.intersection(r2t1.index).intersection(r1t2.index)
    common_features = r1t1.columns.intersection(r2t1.columns).intersection(r1t2.columns)
    rows = []
    for feature in common_features:
        inter = pd.concat([r1t1.loc[common_subjects, feature],
                           r2t1.loc[common_subjects, feature]], axis=1).dropna()
        intra = pd.concat([r1t1.loc[common_subjects, feature],
                           r1t2.loc[common_subjects, feature]], axis=1).dropna()
        inter_icc = icc(inter.to_numpy(), "icc2") if len(inter) >= 2 else np.nan
        intra_icc = icc(intra.to_numpy(), "icc3") if len(intra) >= 2 else np.nan
        rows.append({"feature": feature, "interobserver_icc_2_1": inter_icc,
                     "intraobserver_icc_3_1": intra_icc,
                     "passed": bool(inter_icc >= args.threshold and
                                    intra_icc >= args.threshold)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
