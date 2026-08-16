"""Click commands for the raruma CLI.

Thin wrappers that pass parsed options through to ``scripts``,
following rail_base's ``commands.py`` / ``scripts.py`` split and its
command-group style (``raruma study <subcommand>``).
"""

from typing import Any

import click

from . import options, scripts


@click.group(name="raruma")
# setuptools_scm writes the version into the installed distribution's metadata
@click.version_option(package_name="pz-rail-raruma")
def cli() -> None:
    """raruma analysis scripts"""


@cli.group(name="study")
def study_group() -> None:
    """Degradation-study commands"""


@study_group.command(name="run")
@options.train_file()
@options.test_file()
@options.outdir()
@options.catalog_tag()
@options.informer_class()
@options.estimator_class()
@options.informer_param()
@options.estimator_param()
@options.degradation_types()
@options.noise_grid_start()
@options.noise_grid_stop()
@options.noise_grid_n()
@options.z_grid_max()
@options.z_grid_n()
@options.stats_z_max()
@options.stats_n_bins()
@options.frac()
@options.mag_cutoff()
@options.use_rail_stats()
@options.seed()
@options.n_select()
@options.deviation_threshold()
@options.plot_name_template()
@options.mag_template()
@options.bands()
@options.ref_band()
def run(**kwargs: Any) -> int:
    """Run a degradation study from the command line and save all plots"""
    return scripts.run_study(scripts.StudySettings.from_cli(**kwargs))
