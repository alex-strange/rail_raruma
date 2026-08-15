"""Plots for the photo-z degradation and uncertainty studies.

Library form of the plotting cells of ``nb/Syst/Inform_Estimate.ipynb``:
true-versus-estimated redshift maps, binned performance statistics,
residuals versus magnitude, and the uncertainty-ratio-versus-noise
curve.  Scatter-style plots can overlay density contours showing where
the bulk of the data lies, because a saturated scatter cloud hides the
core of the distribution exactly where most galaxies are.

Also provides the systematic plot-file naming convention used by the
``raruma`` command line:  ``{plot_type}_{param_name}_{param_value}.png``,
for example ``residuals_noise_0.05.png``.

The three functions here that replicate the rows of Fig. 9 of
SITCOMTN-154 (Charles et al., https://sitcomtn-154.lsst.io) take a
``use_rail_plotters`` toggle.  Left True (the default) each delegates to
the official RAIL plotter that drew that row of the technote's figure;
set False it draws the library's own version.  The plotters are from
``rail.plotting.pz_plotters`` in rail_projects, the framework the
technote names as its own plotting and bookkeeping software:
``PZPlotterPointEstimateVsTrueHist2D`` (its
``zestimate_v_ztrue_hist2d_*.png`` files, top row),
``PZPlotterBiweightStatsVsRedshift`` (``biweight_stats_v_redshift_*.png``,
middle row), and ``PZPlotterBiweightStatsVsMag``
(``biweight_stats_v_mag_*.png``, bottom row).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from rail.plotting import pz_plotters

from . import uncertainty_functions as raruma_unc

DEFAULT_PLOT_NAME_TEMPLATE = "{plot_type}_{param_name}_{param_value}.png"

# Fractions of the data mass the density contours enclose: the
# 2D-Gaussian equivalents of 1 and 2 sigma regions
DEFAULT_MASS_FRACTIONS = (0.68, 0.95)

# The official plotters share one input dataset, which validates a
# magnitude field even for the two plots that never read one; an empty
# array satisfies that check without inventing magnitudes
_NO_MAGNITUDES = np.empty(0)


def _rail_plotter_figure(
    plotter_class: type,
    plotter_name: str,
    plotter_config: dict,
    **plot_data: np.ndarray,
) -> Figure:
    """Draw one figure with an official RAIL plotter

    Parameters
    ----------
    plotter_class:
        Plotter class from ``rail.plotting.pz_plotters``

    plotter_name:
        Name of the plot, using rail_projects' own name for it

    plotter_config:
        Plotter configuration, e.g. axis limits and bin counts

    plot_data:
        Data fields the plotter's input dataset requires

    Returns
    -------
    The figure the plotter drew, exactly as RAIL draws it
    """
    plotter = plotter_class(name=plotter_name, **plotter_config)
    # These plotters each make one plot, so unpacking checks that
    (plot_holder,) = plotter.run("", **plot_data).values()
    return plot_holder.figure


def _rail_redshift_binning(z_bins: np.ndarray) -> dict:
    """Express redshift bin edges as the official plotter's binning config

    Parameters
    ----------
    z_bins:
        Bin edges, which must be evenly spaced: the official plotter
        bins with ``np.linspace``, so it takes a range and a count
        rather than edges and cannot reproduce uneven ones

    Returns
    -------
    Configuration giving the same edges, for
    ``PZPlotterBiweightStatsVsRedshift``
    """
    z_bins = np.asarray(z_bins, dtype=float)
    spacings = np.diff(z_bins)
    if not np.allclose(spacings, spacings[:1]):
        raise ValueError(
            "The official plotter bins by (z_min, z_max, count), so it can only draw "
            f"evenly spaced z_bins; got edges {z_bins} with spacings {spacings}"
        )
    return dict(z_min=float(z_bins[0]), z_max=float(z_bins[-1]), n_zbins=len(z_bins))


def format_param_value(param_value: object) -> str:
    """Format a parameter value for use in a plot file name

    Parameters
    ----------
    param_value:
        Value to format; floats are shortened to 4 significant digits
        so noise levels like 0.01778279410038923 stay readable

    Returns
    -------
    String form of the value
    """
    if isinstance(param_value, float):
        return f"{param_value:.4g}"
    return str(param_value)


def plot_filename(
    plot_type: str,
    param_name: str,
    param_value: object,
    template: str = DEFAULT_PLOT_NAME_TEMPLATE,
) -> str:
    """Build a systematic plot file name

    Parameters
    ----------
    plot_type:
        Kind of plot, e.g. 'residuals' or 'statistics'

    param_name:
        Name of the varied parameter, e.g. 'noise' or 'degradation'

    param_value:
        Value of the varied parameter, e.g. 0.05 or 'magnitude_mixed'

    template:
        Naming template with {plot_type}, {param_name}, {param_value}
        placeholders

    Returns
    -------
    File name, e.g. 'residuals_noise_0.05.png'
    """
    return template.format(
        plot_type=plot_type,
        param_name=param_name,
        param_value=format_param_value(param_value),
    )


def _density_contour_levels(hist: np.ndarray, mass_fractions: tuple) -> list[float]:
    """Histogram density thresholds enclosing the given data-mass fractions"""
    sorted_densities = np.sort(hist.ravel())[::-1]
    enclosed_mass = np.cumsum(sorted_densities)
    total_mass = enclosed_mass[-1]
    levels = []
    # Larger fractions need lower thresholds, so this loop yields
    # ascending levels as matplotlib's contour requires
    for fraction in sorted(mass_fractions, reverse=True):
        index = min(
            int(np.searchsorted(enclosed_mass, fraction * total_mass)),
            len(sorted_densities) - 1,
        )
        level = float(sorted_densities[index])
        # Coarse histograms can map two fractions to one threshold;
        # keep levels strictly increasing by nudging duplicates
        if levels and level <= levels[-1]:
            level = np.nextafter(levels[-1], np.inf)
        levels.append(level)
    return levels


def overlay_density_contours(
    axes: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    mass_fractions: tuple = DEFAULT_MASS_FRACTIONS,
    n_bins: int = 50,
    color: str = "black",
    linewidths: float = 1.0,
) -> QuadContourSet:
    """Overlay contours showing where the bulk of the data lies

    Parameters
    ----------
    axes:
        Matplotlib axes to draw on

    x_vals:
        Horizontal values; non-finite entries are ignored

    y_vals:
        Vertical values; non-finite entries are ignored

    mass_fractions:
        Fractions of the (finite) data each contour encloses

    n_bins:
        Number of histogram bins per axis used to estimate the density

    color:
        Contour line color

    linewidths:
        Contour line width

    Returns
    -------
    The drawn contour set
    """
    x_flat = np.asarray(x_vals).ravel()
    y_flat = np.asarray(y_vals).ravel()
    finite = np.isfinite(x_flat) & np.isfinite(y_flat)
    n_finite = int(finite.sum())
    if n_finite < 10:
        raise ValueError(
            f"Need at least 10 finite (x, y) points to draw density contours, got {n_finite}"
        )

    hist, x_edges, y_edges = np.histogram2d(x_flat[finite], y_flat[finite], bins=n_bins)
    levels = _density_contour_levels(hist, mass_fractions)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return axes.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=color,
        linewidths=linewidths,
    )


def plot_true_vs_estimated(
    z_true: np.ndarray,
    z_est: np.ndarray,
    title: str = "",
    z_max: float = 3.0,
    show_contours: bool = True,
    use_rail_plotters: bool = True,
) -> Figure:
    """Plot estimated versus true redshift as a 2D histogram

    Parameters
    ----------
    z_true:
        True (spectroscopic) redshifts

    z_est:
        Estimated redshifts

    title:
        Plot title

    z_max:
        Upper edge of both axes

    show_contours:
        Overlay bulk-of-data density contours

    use_rail_plotters:
        Draw the official version of this panel, the top row of
        SITCOMTN-154 Fig. 9, with RAIL's
        ``PZPlotterPointEstimateVsTrueHist2D``; the default, set False
        for the library's own.  That plotter draws its own labels and
        its own statistics legend, so the title and contour options do
        not apply to it

    Returns
    -------
    Figure with the requested plot
    """
    z_true_flat = np.asarray(z_true).ravel()
    z_est_flat = np.asarray(z_est).ravel()
    if use_rail_plotters:
        return _rail_plotter_figure(
            pz_plotters.PZPlotterPointEstimateVsTrueHist2D,
            "zestimate_v_ztrue_hist2d",
            dict(z_max=z_max),
            truth=z_true_flat,
            pointEstimate=z_est_flat,
            magnitude=_NO_MAGNITUDES,
        )

    figure, axes = plt.subplots(figsize=(8, 6))
    bin_edges = np.linspace(0.0, z_max, 101)
    histogram = axes.hist2d(z_true_flat, z_est_flat, bins=(bin_edges, bin_edges), norm="log")
    z_range = np.linspace(0.0, z_max, 100)
    axes.plot(z_range, z_range, "r--", linewidth=1)
    axes.plot(z_range, z_range + 0.1, "r--", linewidth=1)
    axes.plot(z_range, z_range - 0.1, "r--", linewidth=1)
    if show_contours:
        overlay_density_contours(axes, z_true_flat, z_est_flat, color="white")
    figure.colorbar(histogram[3], ax=axes, label="Density")
    axes.set_xlabel("True Redshift")
    axes.set_ylabel("Estimated Redshift")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def plot_statistics_vs_redshift(
    z_spec: np.ndarray,
    z_phot: np.ndarray,
    z_bins: np.ndarray | None = None,
    use_rail_stats: bool = True,
    title: str = "",
    use_rail_plotters: bool = True,
) -> Figure:
    """Plot bias, scatter, and outlier rate binned by redshift

    Parameters
    ----------
    z_spec:
        Spectroscopic (true) redshifts

    z_phot:
        Estimated redshifts

    z_bins:
        Bin edges; default ``np.linspace(0, 3, 15)`` as in the notebook

    use_rail_stats:
        Statistics mode, see
        ``uncertainty_functions.calc_binned_metrics``

    title:
        Plot title

    use_rail_plotters:
        Draw the official version of this panel, the middle row of
        SITCOMTN-154 Fig. 9, with RAIL's
        ``PZPlotterBiweightStatsVsRedshift``; the default, set False for
        the library's own.  That plotter draws its own labels and always
        uses the robust biweight statistics, so the title and
        statistics-mode options do not apply to it, and it needs evenly
        spaced ``z_bins``

    Returns
    -------
    Figure with the requested plot
    """
    if z_bins is None:
        z_bins = np.linspace(0.0, 3.0, 15)
    if use_rail_plotters:
        return _rail_plotter_figure(
            pz_plotters.PZPlotterBiweightStatsVsRedshift,
            "biweight_stats_v_redshift",
            _rail_redshift_binning(z_bins),
            truth=np.asarray(z_spec).ravel(),
            pointEstimate=np.asarray(z_phot).ravel(),
            magnitude=_NO_MAGNITUDES,
        )

    bin_centers, sigma, bias, outlier_rate = raruma_unc.calc_binned_metrics(
        np.asarray(z_spec).ravel(),
        np.asarray(z_phot).ravel(),
        z_bins,
        use_rail_stats=use_rail_stats,
    )
    figure, axes = plt.subplots(figsize=(8, 6))
    axes.plot(bin_centers, bias, "o-", label="Bias", linewidth=2)
    axes.plot(bin_centers, sigma, "o-", label="Sigma (scatter)", linewidth=2)
    axes.plot(bin_centers, outlier_rate, "o-", label="Outlier rate", linewidth=2)
    axes.set_xlabel("Spectroscopic Redshift")
    axes.set_ylabel("Statistics")
    axes.legend()
    axes.grid(True, alpha=0.3)
    axes.set_title(title)
    figure.tight_layout()
    return figure


def plot_residuals_vs_magnitude(
    magnitudes: np.ndarray,
    z_true: np.ndarray,
    z_est: np.ndarray,
    title: str = "",
    mag_limits: tuple = (18.0, 25.0),
    show_contours: bool = True,
    use_rail_plotters: bool = True,
) -> Figure:
    """Plot normalized redshift residuals against magnitude

    Parameters
    ----------
    magnitudes:
        Reference (i-band) magnitudes of the test galaxies

    z_true:
        True (spectroscopic) redshifts

    z_est:
        Estimated redshifts

    title:
        Plot title

    mag_limits:
        Horizontal axis range

    show_contours:
        Overlay bulk-of-data density contours

    use_rail_plotters:
        Draw the official version of this panel, the bottom row of
        SITCOMTN-154 Fig. 9, with RAIL's ``PZPlotterBiweightStatsVsMag``;
        the default, set False for the library's own.  That plotter adds
        the binned performance metrics above the residuals and draws its
        own labels, so the title and contour options do not apply to it

    Returns
    -------
    Figure with the requested plot
    """
    magnitudes_flat = np.asarray(magnitudes).ravel()
    z_true_flat = np.asarray(z_true).ravel()
    z_est_flat = np.asarray(z_est).ravel()
    if use_rail_plotters:
        return _rail_plotter_figure(
            pz_plotters.PZPlotterBiweightStatsVsMag,
            "biweight_stats_v_magnitude",
            dict(mag_min=mag_limits[0], mag_max=mag_limits[1]),
            truth=z_true_flat,
            pointEstimate=z_est_flat,
            magnitude=magnitudes_flat,
        )

    delta_z = (z_est_flat - z_true_flat) / (1 + z_true_flat)

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.scatter(magnitudes_flat, delta_z, alpha=0.5, s=5)
    axes.axhline(y=0, color="black", linestyle="--", linewidth=1)
    if show_contours:
        overlay_density_contours(axes, magnitudes_flat, delta_z, color="red")
    axes.set_xlabel(r"$i$-band magnitude")
    axes.set_ylabel(r"$(z_{\mathrm{est}} - z_{\mathrm{true}}) \, / \, (1 + z_{\mathrm{true}})$")
    axes.set_xlim(*mag_limits)
    axes.grid(True, alpha=0.3)
    axes.set_title(title)
    figure.tight_layout()
    return figure


def plot_uncertainty_ratio_vs_noise(
    noise_levels: np.ndarray,
    uncertainty_ratios: np.ndarray,
    selected_levels: np.ndarray | None = None,
    title: str = "Uncertainties: Magnitude (One Model Per Noise Level)",
) -> Figure:
    """Plot uncertainty ratios versus noise level

    Parameters
    ----------
    noise_levels:
        Noise levels, one per trained model

    uncertainty_ratios:
        Median uncertainty ratio (degraded / baseline) per noise level

    selected_levels:
        Noise levels picked as interesting (marked with vertical
        lines), e.g. from
        ``uncertainty_functions.select_interesting_noise_levels``

    title:
        Plot title

    Returns
    -------
    Figure with the requested plot
    """
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(noise_levels, uncertainty_ratios, "o-", color="blue", linewidth=2, markersize=6)
    axes.set_xscale("log")
    axes.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Baseline")
    if selected_levels is not None:
        for i, level in enumerate(selected_levels):
            axes.axvline(
                x=level,
                color="green",
                linestyle="--",
                alpha=0.7,
                label="Selected levels" if i == 0 else None,
            )
    axes.set_xlabel("Gaussian Noise Level (mag)")
    axes.set_ylabel("Uncertainty Ratio (noisy / baseline)")
    axes.grid(True, alpha=0.3)
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    return figure
