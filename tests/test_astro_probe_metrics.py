import math

import numpy as np

from mapanything.utils.astro_probe import compute_redshift_metrics, strip_dataset_size_prefix


def test_strip_dataset_size_prefix_handles_resized_expression():
    expr = "4_000 @ AstroSDSSPairDatasetV2(manifest_path='foo.jsonl', split='train')"
    stripped = strip_dataset_size_prefix(expr)
    assert stripped == "AstroSDSSPairDatasetV2(manifest_path='foo.jsonl', split='train')"


def test_compute_redshift_metrics_matches_expected_values():
    z_true = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    z_pred = np.asarray([0.11, 0.18, 0.33], dtype=np.float64)
    metrics = compute_redshift_metrics(z_true, z_pred)

    expected_mae = np.mean(np.abs(z_pred - z_true))
    expected_rmse = np.sqrt(np.mean((z_pred - z_true) ** 2))
    expected_norm = np.abs(z_pred - z_true) / (1.0 + z_true)

    assert math.isclose(metrics["mae_z"], expected_mae, rel_tol=1e-9)
    assert math.isclose(metrics["rmse_z"], expected_rmse, rel_tol=1e-9)
    assert math.isclose(metrics["median_abs_dz_over_1pz"], np.median(expected_norm), rel_tol=1e-9)
    assert math.isclose(metrics["catastrophic_outlier_rate"], 0.0, abs_tol=1e-12)
    assert metrics["pearson_corr"] > 0.9
    assert metrics["spearman_corr"] > 0.9
