"""Tests for rail.raruma.degradation_functions.

Test strategy (subdomains): each degradation type's happy path; the
dispatch table (known key, unknown key); seed reproducibility; and the
contract that the input table is never modified.
"""

import numpy as np
import pytest

degradation_functions = pytest.importorskip(
    "rail.raruma.degradation_functions",
    reason="rail.raruma and its dependencies are required",
)

N_GALAXIES = 1000
SEED = 42


@pytest.fixture(name="catalog")
def fixture_catalog() -> dict[str, np.ndarray]:
    """Synthetic catalog shaped like the DP1 uncertainty-study tables."""
    rng = np.random.default_rng(SEED)
    data = {"redshift": rng.uniform(0.05, 2.5, N_GALAXIES)}
    for band in "ugrizy":
        data[f"{band}_gaap1p0Mag"] = rng.uniform(18.0, 26.0, N_GALAXIES)
    return data


def copy_of(catalog: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: val.copy() for key, val in catalog.items()}


def assert_input_unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    for key, val in before.items():
        np.testing.assert_array_equal(val, after[key])


def test_smear_data_scalar_noise(catalog: dict[str, np.ndarray]) -> None:
    original = copy_of(catalog)
    noisy = degradation_functions.smear_data(catalog, 0.2, seed=SEED)
    differences = noisy["u_gaap1p0Mag"] - original["u_gaap1p0Mag"]
    assert abs(np.std(differences) - 0.2) < 0.05
    assert_input_unchanged(original, catalog)


def test_smear_data_rejects_wrong_length(catalog: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="noise_levels must be scalar or array"):
        degradation_functions.smear_data(catalog, np.array([0.1, 0.2]))


def test_create_mixed_noise_data(catalog: dict[str, np.ndarray]) -> None:
    noise_levels = np.array([0.01, 0.1, 0.5])
    original = copy_of(catalog)
    noisy, per_galaxy_noise = degradation_functions.create_mixed_noise_data(
        catalog, noise_levels, seed=SEED
    )
    assert len(per_galaxy_noise) == N_GALAXIES
    assert set(np.unique(per_galaxy_noise)) <= set(noise_levels)
    differences = noisy["u_gaap1p0Mag"] - original["u_gaap1p0Mag"]
    expected_std = np.sqrt(np.mean(noise_levels**2))
    assert abs(np.std(differences) - expected_std) < 0.1
    assert_input_unchanged(original, catalog)


def test_degrade_data_magnitude_uniform(catalog: dict[str, np.ndarray]) -> None:
    degraded, metadata = degradation_functions.degrade_data(
        catalog, "magnitude_uniform", noise_level=0.2, seed=SEED
    )
    assert metadata == {}
    assert abs(np.std(degraded["u_gaap1p0Mag"] - catalog["u_gaap1p0Mag"]) - 0.2) < 0.05


def test_degrade_data_magnitude_mixed(catalog: dict[str, np.ndarray]) -> None:
    noise_levels = np.array([0.01, 0.1, 0.5])
    _degraded, metadata = degradation_functions.degrade_data(
        catalog, "magnitude_mixed", noise_levels=noise_levels, seed=SEED
    )
    assert len(metadata["per_galaxy_noise"]) == N_GALAXIES


def test_degrade_data_redshift_random(catalog: dict[str, np.ndarray]) -> None:
    degraded, metadata = degradation_functions.degrade_data(
        catalog, "redshift_random", frac=0.05, seed=SEED
    )
    n_changed = int((degraded["redshift"] != catalog["redshift"]).sum())
    assert n_changed == int(0.05 * N_GALAXIES)
    assert len(metadata["degrade_indices"]) == n_changed
    # magnitudes are untouched by a redshift degradation
    np.testing.assert_array_equal(degraded["u_gaap1p0Mag"], catalog["u_gaap1p0Mag"])


def test_degrade_data_redshift_break_formula(catalog: dict[str, np.ndarray]) -> None:
    degraded, metadata = degradation_functions.degrade_data(
        catalog, "redshift_break", frac=0.5, mag_cutoff=24.0, seed=SEED
    )
    indices = metadata["degrade_indices"]
    assert len(indices) > 0
    z_before = catalog["redshift"][indices]
    z_expected = (1 + z_before) * 3646.0 / 1216.0 - 1
    np.testing.assert_allclose(degraded["redshift"][indices], z_expected, rtol=1e-10)
    # every degraded galaxy was faint and above the positivity threshold
    assert (catalog["i_gaap1p0Mag"][indices] > 24.0).all()
    assert (z_before > metadata["zmin"]).all()
    # transformed redshifts stay positive (the zmin guard's whole purpose)
    assert (degraded["redshift"][indices] > 0).all()


def test_degrade_data_redshift_break_no_eligible(catalog: dict[str, np.ndarray]) -> None:
    bright = copy_of(catalog)
    bright["i_gaap1p0Mag"] = np.full(N_GALAXIES, 20.0)
    degraded, metadata = degradation_functions.degrade_data(bright, "redshift_break")
    assert metadata["n_faint"] == 0
    assert len(metadata["degrade_indices"]) == 0
    np.testing.assert_array_equal(degraded["redshift"], bright["redshift"])


def test_degrade_data_unknown_type_raises(catalog: dict[str, np.ndarray]) -> None:
    with pytest.raises(KeyError, match="Unknown degradation_type"):
        degradation_functions.degrade_data(catalog, "not_a_degradation")


def test_dispatch_table_covers_all_types() -> None:
    assert sorted(degradation_functions.DEGRADATION_FUNCS) == [
        "magnitude_mixed",
        "magnitude_uniform",
        "redshift_break",
        "redshift_random",
    ]


@pytest.mark.parametrize(
    "degradation_type,params",
    [
        ("magnitude_uniform", {"noise_level": 0.2}),
        ("magnitude_mixed", {"noise_levels": np.array([0.01, 0.1])}),
        ("redshift_random", {"frac": 0.05}),
        ("redshift_break", {"frac": 0.5}),
    ],
)
def test_seed_reproducibility(
    catalog: dict[str, np.ndarray], degradation_type: str, params: dict
) -> None:
    first, _ = degradation_functions.degrade_data(
        catalog, degradation_type, seed=SEED, **params
    )
    second, _ = degradation_functions.degrade_data(
        catalog, degradation_type, seed=SEED, **params
    )
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
