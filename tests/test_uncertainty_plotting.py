"""Tests for rail.raruma.uncertainty_plotting_functions.

Test strategy (subdomains): the naming convention against the exact
examples in the packaging spec; float formatting; contour overlays on
clean, NaN-bearing, and too-small inputs plus a mass-fraction sanity
check on a known Gaussian; each library figure function returning a
Figure; each official SITCOMTN-154 Fig. 9 plotter (the ``_rail`` half
of each pair) checked against a feature only that drawing has; and the
uneven-bin input the official redshift plotter cannot draw.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

uncertainty_plotting = pytest.importorskip(
    "rail.raruma.uncertainty_plotting_functions",
    reason="rail.raruma and its dependencies are required",
)

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

SEED = 42


def teardown_function() -> None:
    plt.close("all")


def test_plot_filename_matches_spec_examples() -> None:
    assert (
        uncertainty_plotting.plot_filename("residuals", "noise", 0.05)
        == "residuals_noise_0.05.png"
    )
    assert (
        uncertainty_plotting.plot_filename("statistics", "degradation", "magnitude_mixed")
        == "statistics_degradation_magnitude_mixed.png"
    )


def test_plot_filename_user_template_override() -> None:
    name = uncertainty_plotting.plot_filename(
        "residuals", "noise", 0.05, template="{plot_type}--{param_name}--{param_value}.pdf"
    )
    assert name == "residuals--noise--0.05.pdf"


def test_format_param_value_shortens_floats() -> None:
    assert uncertainty_plotting.format_param_value(0.01778279410038923) == "0.01778"
    assert uncertainty_plotting.format_param_value("baseline") == "baseline"


@pytest.fixture(name="gaussian_cloud")
def fixture_gaussian_cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return rng.normal(22.0, 1.0, 5000), rng.normal(0.0, 0.05, 5000)


def test_overlay_density_contours(gaussian_cloud: tuple[np.ndarray, np.ndarray]) -> None:
    x_vals, y_vals = gaussian_cloud
    _figure, axes = plt.subplots()
    contour_set = uncertainty_plotting.overlay_density_contours(axes, x_vals, y_vals)
    assert len(contour_set.levels) == 2
    assert contour_set.levels[0] < contour_set.levels[1]


def test_overlay_density_contours_mass_fractions(
    gaussian_cloud: tuple[np.ndarray, np.ndarray],
) -> None:
    # The 95% contour's density threshold should enclose roughly 95%
    # of the sample (histogram estimate, so the tolerance is loose)
    x_vals, y_vals = gaussian_cloud
    _figure, axes = plt.subplots()
    contour_set = uncertainty_plotting.overlay_density_contours(
        axes, x_vals, y_vals, mass_fractions=(0.95,), n_bins=40
    )
    hist, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=40)
    x_index = np.clip(np.digitize(x_vals, x_edges) - 1, 0, 39)
    y_index = np.clip(np.digitize(y_vals, y_edges) - 1, 0, 39)
    enclosed = hist[x_index, y_index] >= contour_set.levels[0]
    assert 0.85 < enclosed.mean() <= 1.0


def test_overlay_density_contours_ignores_nans(
    gaussian_cloud: tuple[np.ndarray, np.ndarray],
) -> None:
    x_vals, y_vals = gaussian_cloud
    x_with_nans = x_vals.copy()
    x_with_nans[::7] = np.nan  # nondetections
    _figure, axes = plt.subplots()
    contour_set = uncertainty_plotting.overlay_density_contours(axes, x_with_nans, y_vals)
    assert len(contour_set.levels) == 2


def test_overlay_density_contours_rejects_tiny_input() -> None:
    _figure, axes = plt.subplots()
    with pytest.raises(ValueError, match="at least 10 finite"):
        uncertainty_plotting.overlay_density_contours(
            axes, np.array([1.0, np.nan]), np.array([1.0, 2.0])
        )


@pytest.fixture(name="redshift_sample")
def fixture_redshift_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    z_true = rng.uniform(0.05, 2.5, 3000)
    z_est = z_true + rng.normal(0.0, 0.05, 3000) * (1 + z_true)
    magnitudes = rng.uniform(18.0, 25.0, 3000)
    return z_true, z_est, magnitudes


def test_plot_true_vs_estimated(redshift_sample) -> None:
    z_true, z_est, _mags = redshift_sample
    figure = uncertainty_plotting.plot_true_vs_estimated(z_true, z_est, title="Baseline")
    assert isinstance(figure, Figure)
    # The library drawing honors the title it was given
    assert figure.axes[0].get_title() == "Baseline"


def test_plot_statistics_vs_redshift_both_modes(redshift_sample) -> None:
    z_true, z_est, _mags = redshift_sample
    for use_rail_stats in (True, False):
        figure = uncertainty_plotting.plot_statistics_vs_redshift(
            z_true, z_est, use_rail_stats=use_rail_stats, title="Statistics"
        )
        assert isinstance(figure, Figure)
        assert figure.axes[0].get_title() == "Statistics"


def test_plot_residuals_vs_magnitude(redshift_sample) -> None:
    z_true, z_est, magnitudes = redshift_sample
    figure = uncertainty_plotting.plot_residuals_vs_magnitude(
        magnitudes, z_true, z_est, title="Residuals"
    )
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "Residuals"


def test_plot_true_vs_estimated_rail(redshift_sample) -> None:
    z_true, z_est, _mags = redshift_sample
    figure = uncertainty_plotting.plot_true_vs_estimated_rail(z_true, z_est)
    assert isinstance(figure, Figure)
    # The official plotter reports its dz statistics in a legend, ours does not
    assert figure.axes[0].get_legend() is not None


def test_plot_statistics_vs_redshift_rail(redshift_sample) -> None:
    z_true, z_est, _mags = redshift_sample
    figure = uncertainty_plotting.plot_statistics_vs_redshift_rail(z_true, z_est)
    assert isinstance(figure, Figure)
    # The official plotter stacks the metrics over a residual map, ours is one panel
    assert len(figure.axes) == 2


def test_plot_statistics_vs_redshift_rail_rejects_uneven_bins(redshift_sample) -> None:
    z_true, z_est, _mags = redshift_sample
    with pytest.raises(ValueError, match="evenly spaced"):
        uncertainty_plotting.plot_statistics_vs_redshift_rail(
            z_true, z_est, z_bins=np.array([0.0, 0.5, 2.0, 3.0])
        )


def test_plot_residuals_vs_magnitude_rail(redshift_sample) -> None:
    z_true, z_est, magnitudes = redshift_sample
    figure = uncertainty_plotting.plot_residuals_vs_magnitude_rail(magnitudes, z_true, z_est)
    assert isinstance(figure, Figure)
    # The official plotter stacks the metrics over the residual map, ours is one panel
    assert len(figure.axes) == 2


def test_plot_uncertainty_ratio_vs_noise() -> None:
    noise_levels = np.logspace(-4, 0, 17)
    ratios = np.linspace(1.0, 2.0, 17)
    figure = uncertainty_plotting.plot_uncertainty_ratio_vs_noise(
        noise_levels, ratios, selected_levels=noise_levels[8:12]
    )
    assert isinstance(figure, Figure)
