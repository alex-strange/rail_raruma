"""Tests for the raruma CLI.

Test strategy (subdomains): parameter parsing (literals, the special
words, malformed input); class import resolution; help screens for the
group tree; and one full ``raruma study run`` invocation on synthetic
data, checking the exit code, the convention-named plot files, and the
summary json.
"""

import json

import numpy as np
import pytest

scripts = pytest.importorskip(
    "rail.cli.raruma.scripts",
    reason="rail.raruma and its dependencies are required",
)
from click.testing import CliRunner

from rail.cli.raruma.commands import cli

SEED = 42


def test_parse_stage_params() -> None:
    parsed = scripts.parse_stage_params(
        ("nondetect_val=nan", "zmin=0.0", "nzbins=101", "name=knn", "flag=True")
    )
    assert np.isnan(parsed["nondetect_val"])
    assert parsed["zmin"] == 0.0
    assert parsed["nzbins"] == 101
    assert parsed["name"] == "knn"
    assert parsed["flag"] is True


def test_parse_stage_params_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="key=value"):
        scripts.parse_stage_params(("no_equals_sign",))


def test_import_class() -> None:
    numpy_ndarray = scripts.import_class("numpy.ndarray")
    assert numpy_ndarray is np.ndarray
    with pytest.raises(ValueError, match="full 'package.module.Class' path"):
        scripts.import_class("ndarray")


def test_cli_help_screens() -> None:
    runner = CliRunner()
    assert runner.invoke(cli, ["--help"]).exit_code == 0
    assert runner.invoke(cli, ["study", "--help"]).exit_code == 0
    run_help = runner.invoke(cli, ["study", "run", "--help"])
    assert run_help.exit_code == 0
    for option_name in ("--train-file", "--noise-grid-n", "--use-rail-stats", "--seed"):
        assert option_name in run_help.output


def test_study_run_end_to_end(tmp_path) -> None:
    pytest.importorskip("rail.estimation.algos.train_z")
    tables_io = pytest.importorskip("tables_io")

    rng = np.random.default_rng(SEED)
    n_galaxies = 120
    data = {"redshift": rng.uniform(0.1, 2.0, n_galaxies)}
    for band in "ugrizy":
        data[f"{band}_gaap1p0Mag"] = rng.uniform(18.0, 26.0, n_galaxies)
    train_path = tmp_path / "train.hdf5"
    test_path = tmp_path / "test.hdf5"
    tables_io.write(data, str(train_path))
    tables_io.write(data, str(test_path))
    outdir = tmp_path / "study"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "study",
            "run",
            "--train-file", str(train_path),
            "--test-file", str(test_path),
            "--outdir", str(outdir),
            "--catalog-tag", "",
            "--informer-class", "rail.estimation.algos.train_z.TrainZInformer",
            "--estimator-class", "rail.estimation.algos.train_z.TrainZEstimator",
            "--informer-param", "hdf5_groupname=",
            "--informer-param", "zmin=0.0",
            "--informer-param", "zmax=4.0",
            "--informer-param", "nzbins=101",
            "--estimator-param", "hdf5_groupname=",
            "--noise-grid-n", "3",
            "--z-grid-n", "101",
            "--n-select", "2",
            "--seed", str(SEED),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # every flat case gets its three convention-named plots
    for case in ("baseline", "magnitude_mixed", "redshift_random", "redshift_break"):
        for plot_type in ("true_vs_est", "statistics", "residuals"):
            assert (outdir / f"{plot_type}_degradation_{case}.png").exists()
    assert (outdir / "uncertainty_ratio_degradation_magnitude_uniform.png").exists()

    summary = json.loads((outdir / "study_summary.json").read_text())
    assert len(summary["selected_noise_levels"]) == 2
    assert len(summary["noise_levels"]) == 3
    # the selected levels each got their own noise-named plots
    for noise in summary["selected_noise_levels"]:
        noise_tag = scripts.raruma_uplot.format_param_value(float(noise))
        assert (outdir / f"residuals_noise_{noise_tag}.png").exists()
    # per-level model files survive with distinct names
    assert len(list(outdir.glob("magnitude_uniform_*_model.pkl"))) == 3
