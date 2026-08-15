"""Implementations behind the raruma CLI commands.

Kept separate from the click wiring (rail_base's commands/scripts
split) so the same entry points stay importable and testable without a
terminal.
"""

import importlib
import json
import os

import numpy as np
import tables_io
from matplotlib import pyplot as plt

from rail.raruma import degradation_functions as raruma_degrade
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
    param_strings:
        The 'key=value' strings from the command line

    Returns
    -------
    Parsed configuration dict
    """
    import ast

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
    class_path:
        Full import path, e.g.
        'rail.estimation.algos.k_nearneigh.KNearNeighInformer'

    Returns
    -------
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
    catalog_tag:
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


def run_study(
    train_file: str,
    test_file: str,
    outdir: str,
    catalog_tag: str,
    informer_class: str,
    estimator_class: str,
    informer_param: tuple[str, ...],
    estimator_param: tuple[str, ...],
    degradation_types: tuple[str, ...],
    noise_grid_start: float,
    noise_grid_stop: float,
    noise_grid_n: int,
    z_grid_max: float,
    z_grid_n: int,
    stats_z_max: float,
    stats_n_bins: int,
    frac: float,
    mag_cutoff: float,
    use_rail_stats: bool,
    seed: int | None,
    n_select: int,
    deviation_threshold: float,
    plot_name_template: str,
    mag_template: str,
    bands: str,
    ref_band: str,
) -> int:
    """Run a full degradation study and save all plots

    Trains a baseline model plus one model per requested degradation
    case (and one per noise level for 'magnitude_uniform'), computes
    uncertainties and binned metrics, selects the interesting noise
    levels automatically, and writes every plot with the systematic
    naming convention plus a machine-readable summary json.

    Returns
    -------
    0 on success
    """
    os.makedirs(outdir, exist_ok=True)
    apply_catalog_tag(catalog_tag)

    train_data = tables_io.read(train_file)
    test_data = tables_io.read(test_file)

    z_grid = np.linspace(0.0, z_grid_max, z_grid_n)
    noise_grid = np.logspace(noise_grid_start, noise_grid_stop, noise_grid_n)
    stats_z_bins = np.linspace(0.0, stats_z_max, stats_n_bins + 1)

    degradation_params = {
        "magnitude_uniform": {"mag_template": mag_template, "bands": bands},
        "magnitude_mixed": {"mag_template": mag_template, "bands": bands},
        "redshift_random": {"frac": frac},
        "redshift_break": {"frac": frac, "mag_cutoff": mag_cutoff, "mag_template": mag_template},
    }

    print(f"Running degradation study: baseline + {list(degradation_types)}")
    estimates, uncertainties, _metadata = raruma_unc.run_degradation_study(
        train_data,
        test_data,
        z_grid,
        import_class(informer_class),
        import_class(estimator_class),
        informer_kwargs=parse_stage_params(informer_param),
        estimator_kwargs=parse_stage_params(estimator_param),
        degradation_types=degradation_types,
        noise_grid=noise_grid,
        degradation_params=degradation_params,
        seed=seed,
        output_dir=outdir,
    )

    reference_magnitudes = test_data[mag_template.format(band=ref_band)]
    saved_plots = []

    def save_case_plots(case_label: str, estimate, param_name: str, param_value) -> None:
        z_est = np.asarray(estimate.data.ancil["zmode"]).flatten()
        z_true = np.asarray(estimate.data.ancil["redshift"])
        title = CASE_TITLES.get(case_label, case_label)
        if isinstance(param_value, float):
            title = f"Magnitude - Uniform Noise, {param_value:.1e} mag"
        for plot_type, figure in (
            (
                "true_vs_est",
                raruma_uplot.plot_true_vs_estimated(
                    z_true, z_est, title=f"True Z vs. Est. Z ({title})", z_max=stats_z_max
                ),
            ),
            (
                "statistics",
                raruma_uplot.plot_statistics_vs_redshift(
                    z_true,
                    z_est,
                    z_bins=stats_z_bins,
                    use_rail_stats=use_rail_stats,
                    title=f"Statistics vs Redshift ({title})",
                ),
            ),
            (
                "residuals",
                raruma_uplot.plot_residuals_vs_magnitude(
                    reference_magnitudes,
                    z_true,
                    z_est,
                    title=f"Residuals vs Magnitude ({title})",
                ),
            ),
        ):
            filename = raruma_uplot.plot_filename(
                plot_type, param_name, param_value, template=plot_name_template
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
        "train_file": train_file,
        "test_file": test_file,
        "degradation_types": list(degradation_types),
        "seed": seed,
        "use_rail_stats": use_rail_stats,
        "baseline_median_uncertainty": float(np.median(uncertainties["baseline"])),
    }
    if "magnitude_uniform" in degradation_types:
        noise_levels, ratios = raruma_unc.compute_uncertainty_ratios(uncertainties)
        selected_levels = raruma_unc.select_interesting_noise_levels(
            noise_levels, ratios, n_select=n_select, deviation_threshold=deviation_threshold
        )
        figure = raruma_uplot.plot_uncertainty_ratio_vs_noise(
            noise_levels, ratios, selected_levels=selected_levels
        )
        filename = raruma_uplot.plot_filename(
            "uncertainty_ratio", "degradation", "magnitude_uniform", template=plot_name_template
        )
        saved_plots.append(_save_figure(figure, outdir, filename))
        for noise in selected_levels:
            save_case_plots("magnitude_uniform", estimates["magnitude_uniform"][noise], "noise", float(noise))
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
