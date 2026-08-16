"""Tests for the study-level functions in rail.raruma.uncertainty_functions.

Test strategy (subdomains): noise-level selection at the onset of a
rising ratio curve (reproducing the notebook's manual pick), the
no-crossing fallback, the end-of-grid window shift, degenerate sizes,
and bad input; ratio computation from the nested uncertainty dict; and
run_degradation_study end to end (dictionary structure contract,
per-noise-level output files, reproducibility under a fixed seed).
"""

import numpy as np
import pytest

uncertainty_functions = pytest.importorskip(
    "rail.raruma.uncertainty_functions",
    reason="rail.raruma and its dependencies are required",
)

SEED = 42

# The notebook's grid and its manually selected "interesting" levels
NOISE_GRID = np.logspace(-4, 0, 17)
MANUAL_SELECTION = [0.01778279410038923, 0.03162277660168379, 0.05623413251903491, 0.1]


def test_selection_reproduces_manual_pick() -> None:
    # A ratio curve shaped like the notebook's: flat near 1.0 at low
    # noise, deviating past 5% from 0.0178 mag on, rising after that
    ratios = np.ones_like(NOISE_GRID)
    rising = NOISE_GRID >= 0.017
    ratios[rising] = 1.06 + 0.9 * np.log10(NOISE_GRID[rising] / 0.017)
    selected = uncertainty_functions.select_interesting_noise_levels(NOISE_GRID, ratios)
    np.testing.assert_allclose(selected, MANUAL_SELECTION, rtol=1e-12)


def test_selection_counts_low_ratios_as_deviation() -> None:
    # Over-confident models (ratio below 1) are just as interesting
    ratios = np.ones_like(NOISE_GRID)
    ratios[NOISE_GRID >= 0.017] = 0.9
    selected = uncertainty_functions.select_interesting_noise_levels(NOISE_GRID, ratios)
    np.testing.assert_allclose(selected, MANUAL_SELECTION, rtol=1e-12)


def test_selection_falls_back_to_largest_deviations() -> None:
    # Nothing crosses the threshold: pick the largest deviations
    ratios = np.ones_like(NOISE_GRID)
    ratios[[3, 7, 11, 15]] = [1.04, 0.97, 1.03, 1.02]
    selected = uncertainty_functions.select_interesting_noise_levels(NOISE_GRID, ratios)
    np.testing.assert_allclose(selected, NOISE_GRID[[3, 7, 11, 15]], rtol=1e-12)


def test_selection_shifts_window_at_grid_end() -> None:
    # Onset two levels before the end still yields n_select levels
    ratios = np.ones_like(NOISE_GRID)
    ratios[-2:] = 2.0
    selected = uncertainty_functions.select_interesting_noise_levels(NOISE_GRID, ratios)
    np.testing.assert_allclose(selected, NOISE_GRID[-4:], rtol=1e-12)


def test_selection_returns_all_when_grid_small() -> None:
    levels = np.array([0.01, 0.1])
    ratios = np.array([1.0, 1.5])
    selected = uncertainty_functions.select_interesting_noise_levels(levels, ratios)
    np.testing.assert_allclose(selected, levels)


def test_selection_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        uncertainty_functions.select_interesting_noise_levels(
            np.array([0.1, 0.2]), np.array([1.0])
        )


def test_compute_uncertainty_ratios() -> None:
    baseline = np.full((10, 1), 0.2)
    uncertainties = {
        "baseline": baseline,
        "magnitude_uniform": {0.1: baseline * 3.0, 0.01: baseline * 1.5},
    }
    noise_levels, ratios = uncertainty_functions.compute_uncertainty_ratios(uncertainties)
    np.testing.assert_allclose(noise_levels, [0.01, 0.1])
    np.testing.assert_allclose(ratios, [1.5, 3.0])


def test_compute_uncertainty_ratios_missing_key() -> None:
    with pytest.raises(KeyError, match="baseline"):
        uncertainty_functions.compute_uncertainty_ratios({"magnitude_uniform": {}})


def test_run_degradation_study_end_to_end(tmp_path) -> None:
    pytest.importorskip("rail.estimation.algos.train_z")
    from rail.estimation.algos.train_z import TrainZEstimator, TrainZInformer

    rng = np.random.default_rng(SEED)
    n_galaxies = 150
    data = {"redshift": rng.uniform(0.1, 2.0, n_galaxies)}
    for band in "ugrizy":
        data[f"{band}_gaap1p0Mag"] = rng.uniform(18.0, 26.0, n_galaxies)

    z_grid = np.linspace(0.0, 4.0, 101)
    noise_grid = np.array([0.01, 0.1, 0.5])
    stage_kwargs = {"hdf5_groupname": "", "zmin": 0.0, "zmax": 4.0, "nzbins": 101}

    spec = uncertainty_functions.EstimatorSpec(
        informer_class=TrainZInformer,
        estimator_class=TrainZEstimator,
        z_grid=z_grid,
        informer_kwargs=stage_kwargs,
        estimator_kwargs={"hdf5_groupname": ""},
    )
    plan = uncertainty_functions.DegradationPlan(noise_grid=noise_grid, seed=SEED)

    estimates, uncertainties, metadata = uncertainty_functions.run_degradation_study(
        data,
        data,
        spec,
        plan,
        output_dir=str(tmp_path),
    )

    # The notebook's dictionary structure: flat for single cases,
    # nested by noise level for magnitude_uniform
    for case in ("baseline", "magnitude_mixed", "redshift_random", "redshift_break"):
        assert uncertainties[case].shape == (n_galaxies, 1)
    assert sorted(estimates["magnitude_uniform"]) == sorted(noise_grid)
    for noise in noise_grid:
        assert uncertainties["magnitude_uniform"][noise].shape == (n_galaxies, 1)
        # each noise level keeps its own files on disk
        assert (tmp_path / f"magnitude_uniform_{noise:.6g}_output.hdf5").exists()
        assert (tmp_path / f"magnitude_uniform_{noise:.6g}_model.pkl").exists()

    # metadata carries what was degraded
    assert len(metadata["redshift_random"]["train"]["degrade_indices"]) >= 1
    assert "per_galaxy_noise" in metadata["magnitude_mixed"]["train"]

    # ratios flow straight into the selection helper
    noise_levels, ratios = uncertainty_functions.compute_uncertainty_ratios(uncertainties)
    assert len(noise_levels) == len(noise_grid)
    selected = uncertainty_functions.select_interesting_noise_levels(
        noise_levels, ratios, n_select=2
    )
    assert len(selected) == 2

    # a fixed seed reproduces the degradations exactly
    _est2, _unc2, metadata2 = uncertainty_functions.run_degradation_study(
        data,
        data,
        spec,
        plan,
        output_dir=str(tmp_path / "repeat"),
    )
    np.testing.assert_array_equal(
        metadata["redshift_random"]["train"]["degrade_indices"],
        metadata2["redshift_random"]["train"]["degrade_indices"],
    )


def test_run_degradation_study_rejects_unknown_type() -> None:
    with pytest.raises(KeyError, match="Unknown degradation types"):
        uncertainty_functions.run_degradation_study(
            {"redshift": np.array([0.5])},
            {"redshift": np.array([0.5])},
            uncertainty_functions.EstimatorSpec(None, None, np.linspace(0, 4, 11)),
            uncertainty_functions.DegradationPlan(degradation_types=("not_a_type",)),
        )
