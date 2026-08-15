"""Data degradation functions for photo-z uncertainty studies.

Each degradation takes an in-memory table (a dict of numpy arrays, as
returned by ``tables_io.read``), leaves it untouched, and returns a
degraded copy plus a metadata dict describing what was changed.  The
degradations emulate realistic failure modes of training data: extra
photometric noise (uniform or heterogeneous across the sample) and
spectroscopic-redshift errors (random values or systematic Lyman/Balmer
break confusion).

Structure note: RAIL implements spectroscopic degradations as one
``Noisifier``/``Selector`` stage class per effect (see
``rail_astro_tools/src/rail/creation/degraders/spectroscopic_degraders.py``,
``LineConfusion``).  This module keeps the same one-callable-per-effect
decomposition, but as plain functions on in-memory dicts rather than
pipeline stages, because the uncertainty studies run inside notebooks on
data that never round-trips through a ceci pipeline.  The break-confusion
degradation follows ``LineConfusion``'s algorithm (the same ``zmin``
positivity guard and ``(1 + z) * true / wrong - 1`` transform) with the
Lyman/Balmer break pair instead of RAIL's emission-line pairs.
"""

import copy

import numpy as np

from . import admixture_functions as raruma_admix
from . import utility_functions as raruma_util
from .utility_functions import Tablelike

# Magnitude-column naming used by the DP1 uncertainty-study data files.
DEFAULT_MAG_TEMPLATE = "{band}_gaap1p0Mag"
DEFAULT_BANDS = "ugrizy"

# Break wavelengths (Angstroms) for the break-confusion degradation:
# a Balmer break mistaken for a Lyman break shifts the redshift high.
BALMER_BREAK_WAVELEN = 3646.0
LYMAN_BREAK_WAVELEN = 1216.0


def smear_data(
    input_dict: Tablelike,
    noise_levels: float | np.ndarray,
    seed: int | None = None,
    mag_template: str = DEFAULT_MAG_TEMPLATE,
    bands: str = DEFAULT_BANDS,
) -> Tablelike:
    """Add Gaussian noise to magnitude measurements

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    noise_levels:
        Scalar for uniform noise, or array of length n_galaxies
        for per-galaxy noise (in mags)

    seed:
        Random number seed

    mag_template:
        Template for the magnitude column names

    bands:
        Bands to smear

    Returns
    -------
    New dict with noisy magnitudes; the input is not modified
    """
    # Extract magnitude columns into 2D array (n_galaxies, n_bands)
    train_features = raruma_util.get_band_values(input_dict, mag_template, bands)
    n_galaxies = train_features.shape[0]

    # Handle scalar vs per-galaxy noise input

    # If scalar noise_levels => Uniform noise across entire dataset
    if np.isscalar(noise_levels):
        noise_array = noise_levels * np.ones(n_galaxies)

    # If array of length n_galaxies => Apply per-galaxy noise
    elif isinstance(noise_levels, np.ndarray) and len(noise_levels) == n_galaxies:
        noise_array = noise_levels

    # Else (array of other length)
    else:
        raise ValueError(
            f"noise_levels must be scalar or array of length {n_galaxies}, "
            f"got length {len(noise_levels)}"
        )

    # Add Gaussian noise - gaussian_noise expects (n_bands, n_galaxies) input,
    # performs a transpose
    noisy_mags = raruma_admix.gaussian_noise(
        train_features.T, noise_levels=noise_array, seed=seed
    )

    # Create output dictionary (copy input to preserve original)
    output_dict = copy.deepcopy(input_dict)

    # Write noisy magnitudes into output dictionary
    for i, band in enumerate(bands):
        output_dict[mag_template.format(band=band)] = noisy_mags[:, i]

    return output_dict


def create_mixed_noise_data(
    data: Tablelike,
    noise_levels: np.ndarray,
    seed: int | None = None,
    mag_template: str = DEFAULT_MAG_TEMPLATE,
    bands: str = DEFAULT_BANDS,
) -> tuple[Tablelike, np.ndarray]:
    """Create heterogeneous training set with varying noise levels assigned randomly

    Parameters
    ----------
    data:
        Input data (dict from ``tables_io.read``)

    noise_levels:
        Array of noise values to draw from (in mags)

    seed:
        Random number seed

    mag_template:
        Template for the magnitude column names

    bands:
        Bands to smear

    Returns
    -------
    New dict with heterogeneous noise applied, and the array of
    per-galaxy noise levels
    """
    n_galaxies = len(data["redshift"])
    n_levels = len(noise_levels)

    # One seeding governs both the level assignments and the noise draws,
    # so a given seed reproduces the full degradation
    if seed is not None:
        np.random.seed(seed)

    # Randomly assign each galaxy to a noise level
    noise_assignments = np.random.choice(n_levels, size=n_galaxies)

    # Build per-galaxy noise array based on random noise level assignments
    per_galaxy_noise = noise_levels[noise_assignments]

    # Apply noise using smear_data with per-galaxy noise array
    noisy_data = smear_data(data, per_galaxy_noise, mag_template=mag_template, bands=bands)
    return noisy_data, per_galaxy_noise


def degrade_magnitude_uniform(
    input_dict: Tablelike,
    *,
    noise_level: float,
    seed: int | None = None,
    mag_template: str = DEFAULT_MAG_TEMPLATE,
    bands: str = DEFAULT_BANDS,
) -> tuple[Tablelike, dict]:
    """Apply uniform Gaussian magnitude noise to all galaxies

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    noise_level:
        Gaussian noise level applied to every galaxy (in mags)

    seed:
        Random number seed

    mag_template:
        Template for the magnitude column names

    bands:
        Bands to smear

    Returns
    -------
    Degraded copy of the data, and an (empty) metadata dict
    """
    return smear_data(input_dict, noise_level, seed=seed, mag_template=mag_template, bands=bands), {}


def degrade_magnitude_mixed(
    input_dict: Tablelike,
    *,
    noise_levels: np.ndarray,
    seed: int | None = None,
    mag_template: str = DEFAULT_MAG_TEMPLATE,
    bands: str = DEFAULT_BANDS,
) -> tuple[Tablelike, dict]:
    """Apply heterogeneous magnitude noise across the data set

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    noise_levels:
        Array of noise values randomly assigned per galaxy (in mags)

    seed:
        Random number seed

    mag_template:
        Template for the magnitude column names

    bands:
        Bands to smear

    Returns
    -------
    Degraded copy of the data, and metadata with the
    ``per_galaxy_noise`` array
    """
    output_dict, per_galaxy_noise = create_mixed_noise_data(
        input_dict, noise_levels, seed=seed, mag_template=mag_template, bands=bands
    )
    return output_dict, {"per_galaxy_noise": per_galaxy_noise}


def degrade_redshift_random(
    input_dict: Tablelike,
    *,
    frac: float = 0.01,
    seed: int | None = None,
) -> tuple[Tablelike, dict]:
    """Replace a random fraction of redshifts with random values

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    frac:
        Fraction of galaxies to degrade

    seed:
        Random number seed

    Returns
    -------
    Degraded copy of the data, and metadata with the
    ``degrade_indices`` array
    """
    if seed is not None:
        np.random.seed(seed)

    output_dict = copy.deepcopy(input_dict)
    n_galaxies = len(output_dict["redshift"])
    n_degrade = int(frac * n_galaxies)
    degrade_indices = np.random.choice(n_galaxies, n_degrade, replace=False)
    z_min, z_max = output_dict["redshift"].min(), output_dict["redshift"].max()
    output_dict["redshift"][degrade_indices] = np.random.uniform(z_min, z_max, n_degrade)
    return output_dict, {"degrade_indices": degrade_indices}


def degrade_redshift_break(
    input_dict: Tablelike,
    *,
    frac: float = 0.01,
    mag_cutoff: float = 24.0,
    seed: int | None = None,
    mag_template: str = DEFAULT_MAG_TEMPLATE,
) -> tuple[Tablelike, dict]:
    """Apply systematic redshift shifts mimicking Lyman/Balmer break confusion

    Faint galaxies (i-band magnitude above ``mag_cutoff``) have their
    Balmer break (3646 A) misidentified as a Lyman break (1216 A), which
    shifts the recorded redshift high.  Follows the algorithm of RAIL's
    ``LineConfusion`` degrader, including the minimum-redshift guard that
    keeps transformed redshifts positive.

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    frac:
        Fraction of eligible faint galaxies to degrade

    mag_cutoff:
        i-band magnitude above which galaxies are eligible

    seed:
        Random number seed

    mag_template:
        Template for the magnitude column names

    Returns
    -------
    Degraded copy of the data, and metadata with ``degrade_indices``,
    ``n_faint`` (number of eligible galaxies), and ``zmin`` (the
    positivity threshold)
    """
    if seed is not None:
        np.random.seed(seed)

    output_dict = copy.deepcopy(input_dict)

    # Minimum redshift threshold: confusion formula could produce negative
    # (unphysical) redshifts if z_true is too low
    zmin = LYMAN_BREAK_WAVELEN / BALMER_BREAK_WAVELEN - 1

    # Select faint galaxies (i-band magnitude > cutoff) AND above zmin
    # (only degrade galaxies above zmin to ensure transformed redshifts
    # remain positive)
    i_band_column = mag_template.format(band="i")
    faint_mask = (output_dict[i_band_column] > mag_cutoff) & (output_dict["redshift"] > zmin)
    n_faint = faint_mask.sum()

    # If no eligible faint galaxies, return the data unchanged
    if n_faint == 0:
        return output_dict, {"degrade_indices": np.array([]), "n_faint": 0, "zmin": zmin}

    # Select fraction of eligible faint galaxies to degrade
    n_degrade = int(frac * n_faint)
    faint_indices = np.where(faint_mask)[0]
    degrade_indices = np.random.choice(faint_indices, n_degrade, replace=False)

    # Apply break confusion transformation
    z_true = output_dict["redshift"][degrade_indices]
    z_confused = (1 + z_true) * BALMER_BREAK_WAVELEN / LYMAN_BREAK_WAVELEN - 1
    output_dict["redshift"][degrade_indices] = z_confused

    return output_dict, {"degrade_indices": degrade_indices, "n_faint": n_faint, "zmin": zmin}


# Dictionary-based function dispatch: adding a new degradation means adding
# one function and one entry here, with no changes to degrade_data itself
DEGRADATION_FUNCS = dict(
    magnitude_uniform=degrade_magnitude_uniform,
    magnitude_mixed=degrade_magnitude_mixed,
    redshift_random=degrade_redshift_random,
    redshift_break=degrade_redshift_break,
)


def degrade_data(
    input_dict: Tablelike,
    degradation_type: str,
    **params: object,
) -> tuple[Tablelike, dict]:
    """Apply data degradation of the requested type

    Parameters
    ----------
    input_dict:
        Input data (dict from ``tables_io.read``)

    degradation_type:
        One of the keys of ``DEGRADATION_FUNCS``: 'magnitude_uniform',
        'magnitude_mixed', 'redshift_random', 'redshift_break'

    **params:
        Degradation-specific parameters:
        for magnitude_uniform: noise_level (scalar);
        for magnitude_mixed: noise_levels (array);
        for redshift_random: frac (fraction to degrade);
        for redshift_break: frac, mag_cutoff.
        All types also accept seed (random number seed).

    Returns
    -------
    Modified copy with degraded data (original unchanged), and a
    metadata dict describing what was degraded
    """
    try:
        degradation_func = DEGRADATION_FUNCS[degradation_type]
    except KeyError:
        raise KeyError(
            f"Unknown degradation_type: {degradation_type!r}. "
            f"Only {sorted(DEGRADATION_FUNCS)} are allowed"
        ) from None
    return degradation_func(input_dict, **params)
