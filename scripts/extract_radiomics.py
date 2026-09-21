#!/usr/bin/env python3
"""Batch feature extraction adapted from official PyRadiomics examples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def read_and_reorient_lps(path: Union[str, Path]) -> sitk.Image:
    """Read an image and reorient it to the common LPS anatomical frame."""
    image = sitk.ReadImage(str(path))
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image)


def assert_matching_geometry(image: sitk.Image, mask: sitk.Image,
                             patient_id: str) -> None:
    """Stop before extraction when the reoriented image and mask do not align."""
    same = (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-5)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-6)
    )
    if not same:
        raise ValueError(
            f"Image and mask geometry do not match after LPS reorientation for "
            f"{patient_id}. Register/resample the mask to the CT before extraction."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"patient_id", "image_path", "mask_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")

    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.params))
    rows = []
    for case in manifest.itertuples(index=False):
        image = read_and_reorient_lps(case.image_path)
        mask = read_and_reorient_lps(case.mask_path)
        assert_matching_geometry(image, mask, str(case.patient_id))
        result = extractor.execute(image, mask)
        row = {"patient_id": case.patient_id}
        row.update({k: v for k, v in result.items() if not k.startswith("diagnostics_")})
        feature_names = [key for key in row if key != "patient_id"]
        first_order = sum("_firstorder_" in key for key in feature_names)
        shape = sum("_shape_" in key for key in feature_names)
        texture = len(feature_names) - first_order - shape
        observed = (first_order, shape, texture, len(feature_names))
        expected = (360, 14, 1460, 1834)
        if observed != expected:
            raise RuntimeError(
                f"Feature count for {case.patient_id} was {observed}; "
                f"expected {expected} as first-order, shape, texture, and total."
            )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
