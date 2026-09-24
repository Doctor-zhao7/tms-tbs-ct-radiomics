from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from calculate_icc import icc
from run_analysis import (clinical_model, combined_model, delong_auc_ci,
                          delong_pairwise, strict_reference_mask,
                          audit_s10_names)
from run_validation import make_nested_candidate


class CoreStatisticsTests(unittest.TestCase):
    def test_icc_perfect_agreement(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.assertTrue(np.isclose(icc(values, "icc2"), 1.0))
        self.assertTrue(np.isclose(icc(values, "icc3"), 1.0))

    def test_delong_auc_and_pairwise(self):
        outcome = np.array([0, 0, 0, 1, 1, 1])
        first = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        second = np.array([0.2, 0.3, 0.5, 0.5, 0.7, 0.8])
        auc, lower, upper = delong_auc_ci(outcome, first)
        self.assertTrue(np.isclose(auc, 1.0))
        self.assertLessEqual(lower, auc)
        self.assertLessEqual(auc, upper)
        auc1, auc2, p_value = delong_pairwise(outcome, first, second)
        self.assertTrue(np.isclose(auc1, 1.0))
        self.assertGreaterEqual(auc2, 0.0)
        self.assertLessEqual(auc2, 1.0)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_clinical_and_radiomics_inputs_are_standardised(self):
        outcome = np.array([0, 1] * 20)
        data = pd.DataFrame({
            "WBC": np.linspace(3, 25, 40),
            "Hb": np.linspace(70, 150, 40),
            "CRP": np.linspace(1, 100, 40),
            "GLB": np.linspace(20, 60, 40),
            "VAS": np.tile([2, 3, 4, 5], 10),
            "spinal_tenderness": np.tile([0, 1], 20),
            "rad_1": outcome + np.linspace(-0.2, 0.2, 40),
            "rad_2": np.linspace(-2, 2, 40),
        })
        clinical = ["WBC", "Hb", "CRP", "GLB", "VAS",
                    "spinal_tenderness"]
        clinical_estimator = clinical_model(42).fit(data[clinical], outcome)
        self.assertIn("scale", clinical_estimator.named_steps)
        transformed = clinical_estimator[:-1].transform(data[clinical])
        self.assertTrue(np.allclose(transformed.mean(axis=0), 0.0, atol=1e-10))

        combined = combined_model(42, clinical, ["rad_1", "rad_2"])
        combined.fit(data, outcome)
        transformed = combined.named_steps["preprocessing"].transform(data)
        self.assertTrue(np.allclose(transformed.mean(axis=0), 0.0, atol=1e-10))

    def test_manuscript_locked_settings(self):
        config_path = SCRIPT_DIR.parent / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["validation"]["nested_cv_repeats"], 5)
        self.assertEqual(config["feature_selection"]["lasso_cv_folds"], 10)
        self.assertEqual(config["validation"]["strict_reference_values"],
                         ["culture", "targeted_molecular", "mngs"])

        self.assertEqual(
            [config["models"][key] for key in
             ["clinical_threshold", "radiomics_threshold", "combined_threshold"]],
            [0.47, 0.52, 0.49])
        self.assertEqual(config["models"]["unified_svm"]["clinical"],
                         {"C": 1.0, "gamma": "scale", "threshold": 0.46})

        mlp = clinical_model(42).named_steps["mlp"]
        self.assertEqual(mlp.hidden_layer_sizes, (16, 8))
        self.assertEqual(mlp.validation_fraction, 0.20)
        self.assertEqual(mlp.n_iter_no_change, 50)

        fs = config["feature_selection"]
        _, svm_grid = make_nested_candidate(
            "svm", 42, "clinical", config["clinical_features"], [], fs)
        self.assertEqual(svm_grid["model__gamma"],
                         ["scale", 0.001, 0.01, 0.1])
        self.assertEqual(svm_grid["model__class_weight"], [None, "balanced"])
        _, mlp_grid = make_nested_candidate(
            "mlp", 42, "clinical", config["clinical_features"], [], fs)
        self.assertEqual(mlp_grid["model__hidden_layer_sizes"],
                         [(8,), (16,), (8, 4), (16, 8)])
        self.assertEqual(mlp_grid["model__learning_rate_init"],
                         [0.0005, 0.001, 0.005])
        self.assertEqual(mlp_grid["model__batch_size"], [16, 32])

    def test_strict_reference_includes_mngs_only_and_excludes_pathology_only(self):
        evidence = pd.DataFrame({
            "culture_positive": [0, 0, 1, 0],
            "targeted_molecular_positive": [0, 0, 0, 0],
            "mngs_positive": [1, 0, 0, 0],
            "reference_standard": ["mNGS only", "pathology only",
                                   "culture + pathology", "mNGS negative"],
        })
        self.assertEqual(strict_reference_mask(evidence).tolist(),
                         [True, False, True, False])

    def test_strict_reference_rejects_unknown_evidence(self):
        evidence = pd.DataFrame({"culture_positive": [0],
                                 "targeted_molecular_positive": [0],
                                 "mngs_positive": [None]})
        with self.assertRaisesRegex(ValueError, "unknown is not a negative"):
            strict_reference_mask(evidence)

    def test_s10_name_audit_does_not_replace_selected_columns(self):
        comparison = audit_s10_names(
            ["lbp-3D-k_firstorder_Skewness", "wavelet-LHH_glcm_MaximumProbability"],
            ["lbp_3D_k_firstorder_Skewness", "another_feature"])
        self.assertEqual(comparison["matched"].tolist(), [True, False, False])
        self.assertEqual(comparison.iloc[0]["selected_raw_feature"],
                         "lbp-3D-k_firstorder_Skewness")


if __name__ == "__main__":
    unittest.main()
