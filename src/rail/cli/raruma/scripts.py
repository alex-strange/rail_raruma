"""Implementations behind the raruma CLI commands.

Kept separate from the click wiring (rail_base's commands/scripts
split) so the same entry points stay importable and testable without a
terminal.  The command line's flat options arrive here grouped as a
``StudySettings``, built by ``StudySettings.from_cli``.
"""

import ast
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import tables_io
from matplotlib import pyplot as plt

from rail.raruma import uncertainty_functions as raruma_unc
from rail.raruma import uncertainty_plotting_functions as raruma_uplot
from rail.utils import catalog_utils

# Human-readable plot titles, matching the Inform_Estimate notebook
CASE_TITLES = {
    "baseline": "Baseline",
    "magnitude_mixed": "Magnitude - Mixed Noise",
    "redshift_random": "Redshift - Random Noise",
    "redshift_break": "Redshift - Break Confusion",
}

# Literal strings the parameter parser maps to python values
_LITERAL_VALUES = {
    "nan": float("nan"),
    "none": None,
    "true": True,
    "false": False,
}


def parse_stage_params(param_strings: tuple[str, ...]) -> dict:
    """Convert 'key=value' strings into stage configuration values

    Values parse as python literals where possible ('0.1', '[1, 2]'),
    with 'nan', 'none', 'true', and 'false' (any case) mapped to their
    python values; anything else stays a string.

    Parameters
    ----------
    param_strings : tuple[str, ...]
        The 'key=value' strings from the command line

    Returns
    -------
    dict
        Parsed configuration dict
    """
    parsed = {}
    for param_string in param_strings:
        key, separator, raw_value = param_string.partition("=")
        if not separator or not key:
            raise ValueError(f"Poorly formed parameter {param_string!r}. Should be key=value")
        if raw_value.lower() in _LITERAL_VALUES:
            parsed[key] = _LITERAL_VALUES[raw_value.lower()]
            continue
        try:
            parsed[key] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            parsed[key] = raw_value
    return parsed


def import_class(class_path: str) -> type:
    """Import a class from its full dotted path

    Parameters
    ----------
    class_path : str
        Full import path, e.g.
        'rail.estimation.algos.k_nearneigh.KNearNeighInformer'

    Returns
    -------
    type
        The imported class
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(f"{class_path!r} is not a full 'package.module.Class' path")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def apply_catalog_tag(catalog_tag: str) -> None:
    """Apply a RAIL catalog tag so stages pick up the right column names

    Parameters
    ----------
    catalog_tag : str
        Tag name, e.g. 'com_cam_gaap'; empty or None skips
    """
    if not catalog_tag:
        return
    catalog_utils.apply_defaults(catalog_tag)


def _save_figure(figure, outdir: str, filename: str) -> str:
    path = os.path.join(outdir, filename)
    figure.savefig(path)
    plt.close(figure)
    return path


def _summary_ready(value):
    """Make numpy containers json-serializable"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class StudyDataOptions:
    """Where the study reads its data and writes its results"""

    train_file: str
    test_file: str
    outdir: str
    catalog_tag: str


@dataclass
class StudyStageOptions:
    """Which RAIL stages the study trains, and how they are configured"""

    informer_class: str
    estimator_class: str
    informer_param: tuple[str, ...]
    estimator_param: tuple[str, ...]


@dataclass
class StudyGridOptions:
    """The noise, redshift and statistics grids the study sweeps"""

    noise_grid_start: float
    noise_grid_stop: float
    noise_grid_n: int
    z_grid_max: float
    z_grid_n: int
    stats_z_max: float
    stats_n_bins: int


@dataclass
class StudyCaseOptions:
    """Which degradation cases run, and the parameters they degrade with"""

    degradation_types: tuple[str, ...]
    frac: float
    mag_cutoff: float
    seed: int | None


@dataclass
class StudyPlotOptions:
    """What gets plotted, by whom, and under what file names"""

    use_rail_stats: bool
    n_select: int
    deviation_threshold: float
    plot_name_template: str


@dataclass
class StudyColumnOptions:
    """The photometry column names the study reads"""

    mag_template: str
    bands: str
    ref_band: str


@dataclass
class StudySettings:
    """Everything one ``raruma study run`` invocation needs

    The command line supplies two dozen flat options; grouping them here
    keeps every function that carries them down to a handful of
    arguments, and puts each option beside the ones it works with.

    Attributes
    ----------
    data : StudyDataOptions
        Where the study reads its data and writes its results

    stages : StudyStageOptions
        Which RAIL stages the study trains

    grids : StudyGridOptions
        The noise, redshift and statistics grids

    cases : StudyCaseOptions
        Which degradation cases run, and with what parameters

    plots : StudyPlotOptions
        What gets plotted and under what file names

    columns : StudyColumnOptions
        The photometry column names the study reads
    """

    data: StudyDataOptions
    stages: StudyStageOptions
    grids: StudyGridOptions
    cases: StudyCaseOptions
    plots: StudyPlotOptions
    columns: StudyColumnOptions

    @classmethod
    def from_cli(cls, **kwargs: Any) -> "StudySettings":
        """Group the command line's flat options into these six

        Parameters
        ----------
        kwargs : Any
            One entry per ``raruma study run`` option, named as click
            passes it; a missing or unexpected name raises here rather
            than part way into a study

        Returns
        -------
        StudySettings
            The same options, grouped
        """
        settings = cls(
            data=StudyDataOptions(
                train_file=kwargs.pop("train_file"),
                test_file=kwargs.pop("test_file"),
                outdir=kwargs.pop("outdir"),
                catalog_tag=kwargs.pop("catalog_tag"),
            ),
            stages=StudyStageOptions(
                informer_class=kwargs.pop("informer_class"),
                estimator_class=kwargs.pop("estimator_class"),
                informer_param=kwargs.pop("informer_param"),
                estimator_param=kwargs.pop("estimator_param"),
            ),
            grids=StudyGridOptions(
                noise_grid_start=kwargs.pop("noise_grid_start"),
                noise_grid_stop=kwargs.pop("noise_grid_stop"),
                noise_grid_n=kwargs.pop("noise_grid_n"),
                z_grid_max=kwargs.pop("z_grid_max"),
                z_grid_n=kwargs.pop("z_grid_n"),
                stats_z_max=kwargs.pop("stats_z_max"),
                stats_n_bins=kwargs.pop("stats_n_bins"),
            ),
            cases=StudyCaseOptions(
                degradation_types=kwargs.pop("degradation_types"),
                frac=kwargs.pop("frac"),
                mag_cutoff=kwargs.pop("mag_cutoff"),
                seed=kwargs.pop("seed"),
            ),
            plots=StudyPlotOptions(
                use_rail_stats=kwargs.pop("use_rail_stats"),
                n_select=kwargs.pop("n_select"),
                deviation_threshold=kwargs.pop("deviation_threshold"),
                plot_name_template=kwargs.pop("plot_name_template"),
            ),
            columns=StudyColumnOptions(
                mag_template=kwargs.pop("mag_template"),
                bands=kwargs.pop("bands"),
                ref_band=kwargs.pop("ref_band"),
            ),
        )
        if kwargs:
            raise TypeError(f"Not options of a raruma study: {sorted(kwargs)}")
        return settings


def run_study(settings: StudySettings) -> int:
    """Run a full degradation study and save all plots

    Trains a baseline model plus one model per requested degradation
    case (and one per noise level for 'magnitude_uniform'), computes
    uncertainties and binned metrics, selects the interesting noise
    levels automatically, and writes every plot with the systematic
    naming convention plus a machine-readable summary json.

    The three Fig. 9 plots come from the official RAIL plotters, which
    draw their own titles.  ``--no-use-rail-stats`` swaps the statistics
    panel for the library's own drawing, computed with plain numpy
    statistics and carrying the case title.

    Parameters
    ----------
    settings : StudySettings
        Every option the study runs with, grouped into data, stages,
        grids, cases, plots and columns; the command line builds it with
        ``StudySettings.from_cli``

    Returns
    -------
    int
        0 on success
    """
    data, stages, grids = settings.data, settings.stages, settings.grids
    cases, plots, columns = settings.cases, settings.plots, settings.columns
    outdir = data.outdir
    os.makedirs(outdir, exist_ok=True)
    apply_catalog_tag(data.catalog_tag)

    train_data = tables_io.read(data.train_file)
    test_data = tables_io.read(data.test_file)

    z_grid = np.linspace(0.0, grids.z_grid_max, grids.z_grid_n)
    noise_grid = np.logspace(grids.noise_grid_start, grids.noise_grid_stop, grids.noise_grid_n)
    stats_z_bins = np.linspace(0.0, grids.stats_z_max, grids.stats_n_bins + 1)

    mag_template = columns.mag_template
    degradation_params = {
        "magnitude_uniform": {"mag_template": mag_template, "bands": columns.bands},
        "magnitude_mixed": {"mag_template": mag_template, "bands": columns.bands},
        "redshift_random": {"frac": cases.frac},
        "redshift_break": {
            "frac": cases.frac,
            "mag_cutoff": cases.mag_cutoff,
            "mag_template": mag_template,
        },
    }

    degradation_types = cases.degradation_types
    print(f"Running degradation study: baseline + {list(degradation_types)}")
    estimates, uncertainties, _metadata = raruma_unc.run_degradation_study(
        train_data,
        test_data,
        raruma_unc.EstimatorSpec(
            informer_class=import_class(stages.informer_class),
            estimator_class=import_class(stages.estimator_class),
            z_grid=z_grid,
            informer_kwargs=parse_stage_params(stages.informer_param),
            estimator_kwargs=parse_stage_params(stages.estimator_param),
        ),
        raruma_unc.DegradationPlan(
            degradation_types=degradation_types,
            noise_grid=noise_grid,
            degradation_params=degradation_params,
            seed=cases.seed,
        ),
        output_dir=outdir,
    )

    reference_magnitudes = test_data[mag_template.format(band=columns.ref_band)]
    saved_plots = []

    def statistics_figure(z_true: np.ndarray, z_est: np.ndarray, title: str):
        """The official statistics panel, or the library's own drawing"""
        if plots.use_rail_stats:
            return raruma_uplot.plot_statistics_vs_redshift_rail(
                z_true, z_est, z_bins=stats_z_bins
            )
        return raruma_uplot.plot_statistics_vs_redshift(
            z_true, z_est, z_bins=stats_z_bins, use_rail_stats=False, title=title
        )

    def save_case_plots(case_label: str, estimate, param_name: str, param_value) -> None:
        z_est = np.asarray(estimate.data.ancil["zmode"]).flatten()
        z_true = np.asarray(estimate.data.ancil["redshift"])
        title = CASE_TITLES.get(case_label, case_label)
        if isinstance(param_value, float):
            title = f"Magnitude - Uniform Noise, {param_value:.1e} mag"
        for plot_type, figure in (
            (
                "true_vs_est",
                raruma_uplot.plot_true_vs_estimated_rail(
                    z_true, z_est, z_max=grids.stats_z_max
                ),
            ),
            (
                "statistics",
                statistics_figure(z_true, z_est, f"Statistics vs Redshift ({title})"),
            ),
            (
                "residuals",
                raruma_uplot.plot_residuals_vs_magnitude_rail(
                    reference_magnitudes, z_true, z_est
                ),
            ),
        ):
            filename = raruma_uplot.plot_filename(
                plot_type, param_name, param_value, template=plots.plot_name_template
            )
            saved_plots.append(_save_figure(figure, outdir, filename))

    # One plot set per flat case (baseline plus each single-model case)
    for case in ("baseline", *degradation_types):
        if case == "magnitude_uniform":
            continue
        save_case_plots(case, estimates[case], "degradation", case)

    # The per-noise-level case: ratio curve, automatic level selection,
    # then one plot set per selected level
    summary: dict = {
        "train_file": data.train_file,
        "test_file": data.test_file,
        "degradation_types": list(degradation_types),
        "seed": cases.seed,
        "use_rail_stats": plots.use_rail_stats,
        "baseline_median_uncertainty": float(np.median(uncertainties["baseline"])),
    }
    if "magnitude_uniform" in degradation_types:
        noise_levels, ratios = raruma_unc.compute_uncertainty_ratios(uncertainties)
        selected_levels = raruma_unc.select_interesting_noise_levels(
            noise_levels,
            ratios,
            n_select=plots.n_select,
            deviation_threshold=plots.deviation_threshold,
        )
        figure = raruma_uplot.plot_uncertainty_ratio_vs_noise(
            noise_levels, ratios, selected_levels=selected_levels
        )
        filename = raruma_uplot.plot_filename(
            "uncertainty_ratio",
            "degradation",
            "magnitude_uniform",
            template=plots.plot_name_template,
        )
        saved_plots.append(_save_figure(figure, outdir, filename))
        for noise in selected_levels:
            save_case_plots(
                "magnitude_uniform", estimates["magnitude_uniform"][noise], "noise", float(noise)
            )
        summary["noise_levels"] = _summary_ready(noise_levels)
        summary["uncertainty_ratios"] = _summary_ready(ratios)
        summary["selected_noise_levels"] = _summary_ready(selected_levels)
        print(f"Selected interesting noise levels: {np.round(selected_levels, 5).tolist()}")

    summary["plots"] = saved_plots
    summary_path = os.path.join(outdir, "study_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Saved {len(saved_plots)} plots and {summary_path}")
    return 0
