"""Command line options for the raruma CLI.

Follows rail_base's CLI conventions: every option is a reusable
``PartialOption`` (from ``rail.cli.rail.options``) so commands compose
them as decorators and stay consistent with the wider RAIL tooling.
Defaults come from the values ``nb/Syst/Inform_Estimate.ipynb`` used.
"""

import click
from rail.cli.rail.options import PartialOption

__all__ = [
    "bands",
    "catalog_tag",
    "degradation_types",
    "deviation_threshold",
    "estimator_class",
    "estimator_param",
    "frac",
    "informer_class",
    "informer_param",
    "mag_cutoff",
    "mag_template",
    "n_select",
    "noise_grid_start",
    "noise_grid_stop",
    "noise_grid_n",
    "outdir",
    "plot_name_template",
    "ref_band",
    "seed",
    "stats_n_bins",
    "stats_z_max",
    "test_file",
    "train_file",
    "use_rail_stats",
    "z_grid_max",
    "z_grid_n",
]


train_file = PartialOption(
    "--train-file",
    type=click.Path(exists=True),
    required=True,
    help="Input training data file (hdf5, read with tables_io)",
)

test_file = PartialOption(
    "--test-file",
    type=click.Path(exists=True),
    required=True,
    help="Input test data file (hdf5, read with tables_io)",
)

outdir = PartialOption(
    "--outdir",
    type=click.Path(),
    default="degradation_study",
    show_default=True,
    help="Output directory for models, estimates, plots, and the summary",
)

catalog_tag = PartialOption(
    "--catalog-tag",
    type=str,
    default="com_cam_gaap",
    show_default=True,
    help="RAIL catalog tag selecting the column naming; empty string skips",
)

informer_class = PartialOption(
    "--informer-class",
    type=str,
    default="rail.estimation.algos.k_nearneigh.KNearNeighInformer",
    show_default=True,
    help="Full import path of the RAIL informer class",
)

estimator_class = PartialOption(
    "--estimator-class",
    type=str,
    default="rail.estimation.algos.k_nearneigh.KNearNeighEstimator",
    show_default=True,
    help="Full import path of the RAIL estimator class",
)

informer_param = PartialOption(
    "--informer-param",
    type=str,
    multiple=True,
    default=("nondetect_val=nan",),
    show_default=True,
    help="Informer stage parameter as key=value; repeatable, replaces the default set",
)

estimator_param = PartialOption(
    "--estimator-param",
    type=str,
    multiple=True,
    help="Estimator stage parameter as key=value; repeatable",
)

degradation_types = PartialOption(
    "--degradation-type",
    "degradation_types",
    type=str,
    multiple=True,
    default=("magnitude_mixed", "magnitude_uniform", "redshift_random", "redshift_break"),
    show_default=True,
    help="Degradation case to run; repeatable (baseline always runs)",
)

noise_grid_start = PartialOption(
    "--noise-grid-start",
    type=float,
    default=-4.0,
    show_default=True,
    help="log10 of the smallest magnitude noise level",
)

noise_grid_stop = PartialOption(
    "--noise-grid-stop",
    type=float,
    default=0.0,
    show_default=True,
    help="log10 of the largest magnitude noise level",
)

noise_grid_n = PartialOption(
    "--noise-grid-n",
    type=int,
    default=17,
    show_default=True,
    help="Number of noise levels in the (log-spaced) grid",
)

z_grid_max = PartialOption(
    "--z-grid-max",
    type=float,
    default=4.0,
    show_default=True,
    help="Upper edge of the redshift grid for PDF evaluation",
)

z_grid_n = PartialOption(
    "--z-grid-n",
    type=int,
    default=401,
    show_default=True,
    help="Number of points in the redshift grid",
)

stats_z_max = PartialOption(
    "--stats-z-max",
    type=float,
    default=3.0,
    show_default=True,
    help="Upper edge of the redshift binning for the statistics plots",
)

stats_n_bins = PartialOption(
    "--stats-n-bins",
    type=int,
    default=14,
    show_default=True,
    help="Number of redshift bins for the statistics plots",
)

frac = PartialOption(
    "--frac",
    type=float,
    default=0.01,
    show_default=True,
    help="Fraction of galaxies degraded in the redshift cases",
)

mag_cutoff = PartialOption(
    "--mag-cutoff",
    type=float,
    default=24.0,
    show_default=True,
    help="i-band magnitude above which break confusion applies",
)

use_rail_stats = PartialOption(
    "--use-rail-stats/--no-use-rail-stats",
    default=True,
    show_default=True,
    help="Use RAIL's robust biweight statistics (with sigma clipping) for binned metrics",
)

seed = PartialOption(
    "--seed",
    type=int,
    default=None,
    help="Base random seed for reproducible degradations; default is nondeterministic",
)

n_select = PartialOption(
    "--n-select",
    type=int,
    default=4,
    show_default=True,
    help="How many interesting noise levels to select for detailed plots",
)

deviation_threshold = PartialOption(
    "--deviation-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum |uncertainty ratio - 1| that counts as interesting",
)

plot_name_template = PartialOption(
    "--plot-name-template",
    type=str,
    default="{plot_type}_{param_name}_{param_value}.png",
    show_default=True,
    help="Template for plot file names",
)

mag_template = PartialOption(
    "--mag-template",
    type=str,
    default="{band}_gaap1p0Mag",
    show_default=True,
    help="Template for the magnitude column names",
)

bands = PartialOption(
    "--bands",
    type=str,
    default="ugrizy",
    show_default=True,
    help="Bands the magnitude degradations smear",
)

ref_band = PartialOption(
    "--ref-band",
    type=str,
    default="i",
    show_default=True,
    help="Reference band for the residuals-versus-magnitude plots",
)
