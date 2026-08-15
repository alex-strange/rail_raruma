"""Tests for rail.raruma.uncertainty_functions.

Test strategy (subdomains): calc_std against analytic Gaussians; binned
metrics on clean data (numpy and RAIL modes agree), on outlier-spiked
data (RAIL mode stays robust, the drop-in verification for Task 2), and
on empty bins; the handle-based wrapper; and train_test_uncertainty end
to end with a real RAIL informer/estimator pair.
"""

import numpy as np
import pytest

uncertainty_functions = pytest.importorskip(
    "rail.raruma.uncertainty_functions",
    reason="rail.raruma and its dependencies are required",
)
qp = pytest.importorskip("qp")

SEED = 42
TRUE_SIGMA = 0.02
OUTLIER_FRACTION = 0.05


def test_calc_std_matches_analytic_gaussians() -> None:
    grid = np.linspace(0.0, 4.0, 401)
    ensemble = qp.Ensemble(
        qp.stats.norm,
        data=dict(loc=np.array([[1.0], [2.0]]), scale=np.array([[0.1], [0.3]])),
    )
    stds = uncertainty_functions.calc_std(ensemble, grid)
    assert stds.shape == (2, 1)
    np.testing.assert_allclose(stds.flatten(), [0.1, 0.3], rtol=1e-3)


@pytest.fixture(name="residual_sample")
def fixture_residual_sample() -> tuple[np.ndarray, np.ndarray]:
    """z_spec spread over one bin, z_phot with Gaussian core + 5% outliers."""
    rng = np.random.default_rng(SEED)
    n_galaxies = 2000
    z_spec = rng.uniform(0.5, 1.0, n_galaxies)
    scatter = rng.normal(0.0, TRUE_SIGMA, n_galaxies)
    n_outliers = int(OUTLIER_FRACTION * n_galaxies)
    scatter[:n_outliers] += 1.0  # catastrophic outliers, all one-sided
    z_phot = z_spec + scatter * (1 + z_spec)
    return z_spec, z_phot


def test_binned_metrics_clean_data_modes_agree() -> None:
    rng = np.random.default_rng(SEED)
    n_galaxies = 5000
    z_spec = rng.uniform(0.5, 1.0, n_galaxies)
    z_phot = z_spec + rng.normal(0.0, TRUE_SIGMA, n_galaxies) * (1 + z_spec)
    z_bins = np.array([0.5, 1.0])
    for use_rail_stats in (True, False):
        _centers, sigma, bias, outlier = uncertainty_functions.calc_binned_metrics(
            z_spec, z_phot, z_bins, use_rail_stats=use_rail_stats
        )
        assert abs(sigma[0] - TRUE_SIGMA) < 0.002
        assert abs(bias[0]) < 0.002
        assert outlier[0] == 0.0


def test_binned_metrics_rail_stats_resist_outliers(
    residual_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    z_spec, z_phot = residual_sample
    z_bins = np.array([0.5, 1.0])

    _c, sigma_np, bias_np, outlier_np = uncertainty_functions.calc_binned_metrics(
        z_spec, z_phot, z_bins, use_rail_stats=False
    )
    _c, sigma_rail, bias_rail, outlier_rail = uncertainty_functions.calc_binned_metrics(
        z_spec, z_phot, z_bins, use_rail_stats=True
    )

    # numpy statistics are dragged by the 5% outlier population
    assert sigma_np[0] > 5 * TRUE_SIGMA
    assert bias_np[0] > 0.02
    # biweight + sigma clipping recover the core distribution
    assert abs(sigma_rail[0] - TRUE_SIGMA) < 0.005
    assert abs(bias_rail[0]) < 0.005
    # the outlier rate is mode-independent and counts the spiked fraction
    assert outlier_np[0] == outlier_rail[0]
    assert abs(outlier_rail[0] - OUTLIER_FRACTION) < 0.01


def test_binned_metrics_empty_bin_is_nan(
    residual_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    z_spec, z_phot = residual_sample
    z_bins = np.array([0.5, 1.0, 3.0])  # second bin is empty
    for use_rail_stats in (True, False):
        _c, sigma, bias, outlier = uncertainty_functions.calc_binned_metrics(
            z_spec, z_phot, z_bins, use_rail_stats=use_rail_stats
        )
        assert np.isnan(sigma[1]) and np.isnan(bias[1]) and np.isnan(outlier[1])
        assert np.isfinite(sigma[0])


class _FakeEnsemble:
    def __init__(self, ancil: dict) -> None:
        self.ancil = ancil


class _FakeHandle:
    def __init__(self, ancil: dict) -> None:
        self.data = _FakeEnsemble(ancil)


def test_handle_wrapper_matches_array_form(
    residual_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    z_spec, z_phot = residual_sample
    z_bins = np.linspace(0.5, 1.0, 4)
    handle = _FakeHandle({"redshift": z_spec, "zmode": z_phot.reshape(-1, 1)})
    from_handle = uncertainty_functions.calc_photoz_performance_metrics(handle, z_bins)
    from_arrays = uncertainty_functions.calc_binned_metrics(z_spec, z_phot, z_bins)
    for got, expected in zip(from_handle, from_arrays):
        np.testing.assert_array_equal(got, expected)


def test_train_test_uncertainty_end_to_end(tmp_path) -> None:
    pytest.importorskip("rail.estimation.algos.train_z")
    from rail.estimation.algos.train_z import TrainZEstimator, TrainZInformer

    rng = np.random.default_rng(SEED)
    n_galaxies = 200
    data = {"redshift": rng.uniform(0.1, 2.0, n_galaxies)}
    for band in "ugrizy":
        data[f"{band}_gaap1p0Mag"] = rng.uniform(18.0, 25.0, n_galaxies)

    z_grid = np.linspace(0.0, 4.0, 401)
    stage_kwargs = {"hdf5_groupname": "", "zmin": 0.0, "zmax": 4.0, "nzbins": 401}
    uncertainty, estimate_handle = uncertainty_functions.train_test_uncertainty(
        data,
        data,
        z_grid,
        TrainZInformer,
        TrainZEstimator,
        informer_kwargs=stage_kwargs,
        estimator_kwargs={"hdf5_groupname": ""},
        label=str(tmp_path / "trainz_smoke"),
    )
    assert uncertainty.shape == (n_galaxies, 1)
    assert np.isfinite(uncertainty).all()
    assert (uncertainty > 0).all()
    ancil = estimate_handle.data.ancil
    assert "zmode" in ancil and "redshift" in ancil
    assert (tmp_path / "trainz_smoke_model.pkl").exists()
    assert (tmp_path / "trainz_smoke_output.hdf5").exists()
    # a second run with the same label must overwrite cleanly (the
    # repeated-cycle scenario reset_data_store exists for)
    uncertainty_repeat, _ = uncertainty_functions.train_test_uncertainty(
        data,
        data,
        z_grid,
        TrainZInformer,
        TrainZEstimator,
        informer_kwargs=stage_kwargs,
        estimator_kwargs={"hdf5_groupname": ""},
        label=str(tmp_path / "trainz_smoke"),
    )
    np.testing.assert_allclose(uncertainty_repeat, uncertainty)
