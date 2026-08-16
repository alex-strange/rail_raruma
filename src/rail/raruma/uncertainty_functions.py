"""Uncertainty quantification for photo-z estimates under data degradation.

Ports the analysis workflow of ``nb/Syst/Inform_Estimate.ipynb`` into
library form: train a RAIL estimator on (possibly degraded) data, extract
per-galaxy p(z) widths as the uncertainty, compute binned performance
metrics, and orchestrate a full degradation study across degradation
types and noise levels.

Metrics can use either plain numpy statistics or the robust estimators
RAIL's plotting tools use (astropy biweight location/scale after
iterative sigma clipping, the pattern of
``rail_projects/src/rail/plotting/pz_plotters.py``); the robust form is
the default because photo-z residual distributions have heavy outlier
tails that drag a plain mean/std away from the core of the distribution.
"""

import os
from dataclasses import dataclass

import numpy as np
import qp
from rail.core.data import DataStore
from rail.core.stage import RailStage

from . import degradation_functions as raruma_degrade
from . import plotting_functions as raruma_plot
from .utility_functions import Tablelike

# |dz/(1+z)| above this counts as a catastrophic outlier (photo-z convention)
OUTLIER_THRESHOLD = 0.15

# The degradation cases studied by the Inform_Estimate notebook, in its order
DEFAULT_DEGRADATION_TYPES = (
    "magnitude_mixed",
    "magnitude_uniform",
    "redshift_random",
    "redshift_break",
)


def reset_data_store() -> None:
    """Clear RAIL's shared DataStore and allow output overwrites

    Older rail_base versions keep one shared ``RailStage.data_store``
    whose stale 'input' entries can silently feed a new stage the
    previous run's data; newer versions give each stage its own store.
    This helper makes repeated in-memory train/test cycles safe on both.
    """
    DataStore.allow_overwrite = True
    data_store = getattr(RailStage, "data_store", None)
    if data_store is not None:
        data_store.clear()


def calc_std(qp_dstn: qp.Ensemble, grid: np.ndarray) -> np.ndarray:
    """Calculate the standard deviation of each redshift PDF

    Parameters
    ----------
    qp_dstn : qp.Ensemble
        qp Ensemble of per-galaxy redshift PDFs

    grid : np.ndarray
        Redshift grid on which to evaluate the PDFs

    Returns
    -------
    np.ndarray
        Standard deviation for each PDF, shape (n_galaxies, 1)
    """
    pdfs = qp_dstn.pdf(grid)  # Get PDF values on grid
    norms = pdfs.sum(axis=1)  # Normalize
    means = np.sum(pdfs * grid, axis=1) / norms  # Calculate mean redshift
    diffs = (np.expand_dims(grid, -1) - means).T  # Distance from mean
    wt_diffs = diffs * diffs * pdfs  # Weighted squared differences
    stds = np.sqrt((wt_diffs).sum(axis=1) / norms)  # Standard deviation
    return np.expand_dims(stds, -1)


def calc_binned_metrics(
    z_spec: np.ndarray,
    z_phot: np.ndarray,
    z_bins: np.ndarray,
    use_rail_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate photo-z performance metrics binned by spectroscopic redshift

    Parameters
    ----------
    z_spec : np.ndarray
        Spectroscopic (true) redshifts

    z_phot : np.ndarray
        Estimated redshifts

    z_bins : np.ndarray
        Bin edges for spectroscopic redshift

    use_rail_stats : bool
        If True (default), bias and sigma are the astropy biweight
        location and scale of each bin after iterative sigma clipping
        (the robust pattern RAIL's pz_plotters uses); if False, plain
        ``np.mean`` and ``np.std``.  The outlier rate is the fraction of
        the FULL bin with ``|dz| > OUTLIER_THRESHOLD`` in both modes, so
        clipped-away catastrophic outliers still count and the metric
        keeps one meaning across modes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        bin_centers, sigma_array, bias_array, outlier_rate_array
        (empty bins hold NaN)
    """
    # Calculate normalized residuals
    delta_z = (z_phot - z_spec) / (1 + z_spec)

    # Initialize output arrays
    n_bins = len(z_bins) - 1
    bin_centers = (z_bins[:-1] + z_bins[1:]) / 2
    sigma_array = np.zeros(n_bins)
    bias_array = np.zeros(n_bins)
    outlier_rate_array = np.zeros(n_bins)

    # Calculate metrics for each bin
    for i in range(n_bins):
        mask = (z_spec >= z_bins[i]) & (z_spec < z_bins[i + 1])
        n_in_bin = mask.sum()

        # Empty bin
        if n_in_bin == 0:
            sigma_array[i] = np.nan
            bias_array[i] = np.nan
            outlier_rate_array[i] = np.nan
            continue

        delta_z_bin = delta_z[mask]
        if use_rail_stats:
            bias, _bias_err, sigma, _rel_outlier = raruma_plot.get_biweight_mean_sigma_outlier(
                delta_z_bin, nclip=3
            )
            bias_array[i] = bias
            sigma_array[i] = sigma
        else:
            sigma_array[i] = np.std(delta_z_bin)
            bias_array[i] = np.mean(delta_z_bin)
        outlier_rate_array[i] = (np.abs(delta_z_bin) > OUTLIER_THRESHOLD).sum() / n_in_bin

    return bin_centers, sigma_array, bias_array, outlier_rate_array


def calc_photoz_performance_metrics(
    estimate_handle,
    z_bins: np.ndarray,
    use_rail_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate photo-z performance metrics for a RAIL estimate

    Parameters
    ----------
    estimate_handle : QPHandle
        Output handle from ``train_test_uncertainty``; its ancillary
        data must carry 'redshift' (true) and 'zmode' (estimate)

    z_bins : np.ndarray
        Bin edges for spectroscopic redshift

    use_rail_stats : bool
        Statistics mode, see ``calc_binned_metrics``

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        bin_centers, sigma_array, bias_array, outlier_rate_array
    """
    # Extract redshifts
    z_spec = estimate_handle.data.ancil["redshift"]
    z_phot = estimate_handle.data.ancil["zmode"].flatten()
    return calc_binned_metrics(z_spec, z_phot, z_bins, use_rail_stats=use_rail_stats)


def train_test_uncertainty(
    train_data: Tablelike,
    test_data: Tablelike,
    z_grid: np.ndarray,
    informer_class,
    estimator_class,
    informer_kwargs: dict | None = None,
    estimator_kwargs: dict | None = None,
    label: str = "output",
):  # pylint: disable=too-many-arguments,too-many-positional-arguments  # every argument used
    """Train a model, test it, and extract uncertainty for any algorithm

    Parameters
    ----------
    train_data : Tablelike
        Training data (dict from ``tables_io.read``)

    test_data : Tablelike
        Test data (dict from ``tables_io.read``)

    z_grid : np.ndarray
        Redshift grid for PDF evaluation

    informer_class : type
        RAIL informer, e.g. ``KNearNeighInformer``

    estimator_class : type
        RAIL estimator, e.g. ``KNearNeighEstimator``

    informer_kwargs : dict | None
        Parameters for the informer stage

    estimator_kwargs : dict | None
        Parameters for the estimator stage

    label : str
        Identifier for the output files ``<label>_model.pkl`` and
        ``<label>_output.hdf5``; may include a directory prefix

    Returns
    -------
    tuple[np.ndarray, QPHandle]
        Uncertainty array of shape (n_test_galaxies, 1), and the estimate
        handle
    """
    if informer_kwargs is None:
        informer_kwargs = {}
    if estimator_kwargs is None:
        estimator_kwargs = {}

    reset_data_store()

    label_dir = os.path.dirname(label)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)

    # Remove any stale model file so a failed inform cannot leave an old
    # model in place to be picked up silently
    model_path = f"{label}_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Train model
    informer = informer_class.make_stage(model=model_path, **informer_kwargs)
    model = informer.inform(train_data)

    # Test model
    estimator = estimator_class.make_stage(
        model=model, output=f"{label}_output.hdf5", **estimator_kwargs
    )
    estimate_handle = estimator.estimate(test_data)

    # Extract PDFs and calculate uncertainty
    ensemble = qp.read(estimate_handle.path)
    uncertainty = calc_std(ensemble, z_grid)

    return uncertainty, estimate_handle


def compute_uncertainty_ratios(
    uncertainties: dict,
    case: str = "magnitude_uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute median uncertainty ratios versus noise level

    Parameters
    ----------
    uncertainties : dict
        Uncertainty dict from ``run_degradation_study``; must hold
        'baseline' and a ``case`` entry keyed by noise level

    case : str
        Which noise-level-keyed case to ratio against baseline

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Noise levels (ascending) and the median per-galaxy ratio of that
        level's uncertainty to the baseline uncertainty
    """
    if "baseline" not in uncertainties or case not in uncertainties:
        raise KeyError(
            f"uncertainties must hold 'baseline' and {case!r}; got {sorted(uncertainties)}"
        )
    noise_levels = np.sort(np.asarray(list(uncertainties[case].keys()), dtype=float))
    ratios = np.array(
        [
            np.median(uncertainties[case][noise] / uncertainties["baseline"])
            for noise in noise_levels
        ]
    )
    return noise_levels, ratios


def select_interesting_noise_levels(
    noise_levels: np.ndarray,
    uncertainty_ratios: np.ndarray,
    n_select: int = 4,
    deviation_threshold: float = 0.05,
) -> np.ndarray:
    """Select representative noise levels from the interesting region

    The interesting region is where the uncertainty ratio deviates from
    1.0 by more than ``deviation_threshold``.  The selection takes
    ``n_select`` consecutive levels starting at the onset of that
    region, because the transition where calibration first degrades is
    the scientifically informative part (far above the onset the model
    is uniformly badly calibrated).  This automates the manual pick the
    Inform_Estimate notebook made from the uncertainty-ratio plot.

    Parameters
    ----------
    noise_levels : np.ndarray
        Noise levels, one per trained model

    uncertainty_ratios : np.ndarray
        Median uncertainty ratio (degraded / baseline) per noise level

    n_select : int
        How many levels to select

    deviation_threshold : float
        Minimum ``|ratio - 1|`` that counts as interesting

    Returns
    -------
    np.ndarray
        Selected noise levels, ascending; if no level crosses the
        threshold, the ``n_select`` levels with the largest deviations
    """
    noise_levels = np.asarray(noise_levels, dtype=float)
    uncertainty_ratios = np.asarray(uncertainty_ratios, dtype=float)
    if noise_levels.shape != uncertainty_ratios.shape:
        raise ValueError(
            f"noise_levels and uncertainty_ratios must have the same shape, "
            f"got {noise_levels.shape} and {uncertainty_ratios.shape}"
        )
    if n_select < 1:
        raise ValueError(f"n_select must be at least 1, got {n_select}")
    if len(noise_levels) <= n_select:
        return np.sort(noise_levels)

    # Work in ascending noise order so "onset" means lowest interesting noise
    order = np.argsort(noise_levels)
    noise_levels = noise_levels[order]
    deviations = np.abs(uncertainty_ratios[order] - 1.0)

    interesting = np.where(deviations > deviation_threshold)[0]
    if len(interesting) == 0:
        # Nothing crosses the threshold: fall back to the largest deviations
        picked = np.sort(np.argsort(deviations)[::-1][:n_select])
        return noise_levels[picked]

    # Consecutive window from the onset, shifted back if it overruns the end
    start = min(interesting[0], len(noise_levels) - n_select)
    return noise_levels[start : start + n_select]


@dataclass
class EstimatorSpec:
    """What every case in a study trains and estimates with

    These five travel together through the whole study: the study hands
    them to ``train_test_uncertainty`` unchanged, once per case.

    Attributes
    ----------
    informer_class : type
        RAIL informer, e.g. ``KNearNeighInformer``

    estimator_class : type
        RAIL estimator, e.g. ``KNearNeighEstimator``

    z_grid : np.ndarray
        Redshift grid for PDF evaluation

    informer_kwargs : dict | None
        Parameters for the informer stage

    estimator_kwargs : dict | None
        Parameters for the estimator stage
    """

    informer_class: type
    estimator_class: type
    z_grid: np.ndarray
    informer_kwargs: dict | None = None
    estimator_kwargs: dict | None = None


@dataclass
class DegradationPlan:
    """Which degradations a study runs, and with what parameters

    Attributes
    ----------
    degradation_types : tuple[str, ...]
        Which degradation cases to run (baseline always runs)

    noise_grid : np.ndarray | None
        Noise levels for the magnitude cases; default
        ``np.logspace(-4, 0, 17)`` as in the notebook

    degradation_params : dict | None
        Optional per-case parameter overrides, e.g.
        ``{'redshift_break': {'frac': 0.02, 'mag_cutoff': 23.5}}``

    seed : int | None
        Base random seed; each degradation call gets its own child seed
        so runs are reproducible but cases stay decorrelated.  None
        (default) keeps the notebook's nondeterministic behavior
    """

    degradation_types: tuple[str, ...] = DEFAULT_DEGRADATION_TYPES
    noise_grid: np.ndarray | None = None
    degradation_params: dict | None = None
    seed: int | None = None


def run_degradation_study(
    train: Tablelike,
    test: Tablelike,
    spec: EstimatorSpec,
    plan: DegradationPlan | None = None,
    output_dir: str = "",
) -> tuple[dict, dict, dict]:
    """Run the full degradation study: baseline plus each degradation case

    Reproduces the train-and-test sweep of the Inform_Estimate notebook.
    'magnitude_uniform' trains one model per entry of ``noise_grid``;
    every other case trains one model on data degraded that one way.
    Train and test are degraded independently with the same parameters,
    as in the notebook.

    Parameters
    ----------
    train : Tablelike
        Training data (dict from ``tables_io.read``)

    test : Tablelike
        Test data (dict from ``tables_io.read``)

    spec : EstimatorSpec
        The stages, grid and stage options every case trains with

    plan : DegradationPlan | None
        Which degradations to run and with what parameters; default
        runs ``DEFAULT_DEGRADATION_TYPES`` on the notebook's noise grid

    output_dir : str
        Directory for the per-case model and estimate files; default
        is the working directory, as in the notebook

    Returns
    -------
    tuple[dict, dict, dict]
        estimates, uncertainties, metadata dicts keyed by case name, with
        the 'magnitude_uniform' entries nested dicts keyed by noise level
        (the notebook's structure, which downstream plotting depends on),
        plus 'baseline' in estimates and uncertainties
    """
    plan = plan or DegradationPlan()
    degradation_types = plan.degradation_types
    noise_grid = plan.noise_grid
    if noise_grid is None:
        noise_grid = np.logspace(-4, 0, 17)
    degradation_params = dict(plan.degradation_params or {})
    unknown_types = set(degradation_types) - set(raruma_degrade.DEGRADATION_FUNCS)
    if unknown_types:
        raise KeyError(
            f"Unknown degradation types {sorted(unknown_types)}. "
            f"Only {sorted(raruma_degrade.DEGRADATION_FUNCS)} are allowed"
        )

    seed_sequence = None if plan.seed is None else np.random.SeedSequence(plan.seed)

    def next_seed() -> int | None:
        if seed_sequence is None:
            return None
        return int(seed_sequence.spawn(1)[0].generate_state(1)[0])

    def run_one(train_data: Tablelike, test_data: Tablelike, label: str):
        return train_test_uncertainty(
            train_data,
            test_data,
            spec.z_grid,
            spec.informer_class,
            spec.estimator_class,
            informer_kwargs=spec.informer_kwargs,
            estimator_kwargs=spec.estimator_kwargs,
            label=os.path.join(output_dir, label),
        )

    estimates: dict = {}
    uncertainties: dict = {}
    metadata: dict = {}

    # Baseline on clean data; every ratio is measured against this
    uncertainties["baseline"], estimates["baseline"] = run_one(train, test, "baseline")

    for degradation_type in degradation_types:
        params = degradation_params.get(degradation_type, {})

        # One model per noise level, each with its own label so every
        # level's model and estimate files stay on disk separately
        if degradation_type == "magnitude_uniform":
            estimates[degradation_type] = {}
            uncertainties[degradation_type] = {}
            metadata[degradation_type] = {}
            for noise in noise_grid:
                train_degraded, meta_train = raruma_degrade.degrade_data(
                    train, degradation_type, noise_level=noise, seed=next_seed(), **params
                )
                test_degraded, meta_test = raruma_degrade.degrade_data(
                    test, degradation_type, noise_level=noise, seed=next_seed(), **params
                )
                unc, est = run_one(
                    train_degraded, test_degraded, f"{degradation_type}_{noise:.6g}"
                )
                estimates[degradation_type][noise] = est
                uncertainties[degradation_type][noise] = unc
                metadata[degradation_type][noise] = {"train": meta_train, "test": meta_test}
            continue

        # The mixed-noise case draws its per-galaxy levels from the grid
        if degradation_type == "magnitude_mixed":
            params = {"noise_levels": noise_grid, **params}

        train_degraded, meta_train = raruma_degrade.degrade_data(
            train, degradation_type, seed=next_seed(), **params
        )
        test_degraded, meta_test = raruma_degrade.degrade_data(
            test, degradation_type, seed=next_seed(), **params
        )
        unc, est = run_one(train_degraded, test_degraded, degradation_type)
        estimates[degradation_type] = est
        uncertainties[degradation_type] = unc
        metadata[degradation_type] = {"train": meta_train, "test": meta_test}

    return estimates, uncertainties, metadata
