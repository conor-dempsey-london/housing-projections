"""
Shared fixtures for the housing-projections test suite.

Synthetic data avoids all file I/O and network calls so tests run offline.
"""
import arviz as az
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from housing_projections.config import (
    ALL_COLS_BEN,
    ALL_COLS_PLAN,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
)
from housing_projections.data import make_data_dict

# ── Constants ──────────────────────────────────────────────────────────────────

N_AREAS = 9   # 3×3 grid — enough for Queen contiguity
N_YEARS = 10  # matches len(INFER_COLS_PLAN)
N_CHAINS = 2
N_DRAWS  = 40


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_idata(posterior: dict, sample_stats: dict | None = None,
               log_likelihood: dict | None = None) -> az.InferenceData:
    """
    Build an az.InferenceData from plain numpy dicts.

    Uses the current arviz API: az.from_dict takes a nested dict where outer
    keys are group names ('posterior', 'sample_stats', …) and inner keys are
    variable names. The first two array dimensions are treated as chain/draw.
    """
    groups: dict = {'posterior': posterior}
    if sample_stats is not None:
        groups['sample_stats'] = sample_stats
    if log_likelihood is not None:
        groups['log_likelihood'] = log_likelihood
    return az.from_dict(groups)


@pytest.fixture(scope='session')
def mock_traces_with_ll(data_dict, rng):
    """
    Two minimal InferenceData objects with log_likelihood groups,
    suitable for testing compute_model_comparison (az.compare).
    """
    n_areas = data_dict['n_areas']
    n_years = data_dict['n_years']
    n_obs   = n_areas * n_years

    traces = {}
    for name in ('MA', 'MB'):
        z = rng.normal(1.0, 2.0, size=(N_CHAINS, N_DRAWS, n_areas, n_years))
        # log_likelihood shape: (chains, draws, n_obs) — one value per observation
        ll = rng.normal(-1.5, 0.3, size=(N_CHAINS, N_DRAWS, n_obs))
        traces[name] = make_idata(
            posterior={'z': z},
            sample_stats={'diverging': np.zeros((N_CHAINS, N_DRAWS), dtype=bool)},
            log_likelihood={'P_like': ll},
        )
    return traces


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope='session')
def synthetic_gdf(rng):
    """
    3×3 grid of 1-unit square LSOAs in EPSG:27700.
    Includes all planning, BEN, and census columns needed by the package.
    Adjacent squares share borders, so Queen contiguity is well-defined.
    """
    rows, geometries = [], []

    for i in range(3):
        for j in range(3):
            geometries.append(box(j, i, j + 1, i + 1))
            row = {
                'LSOA21CD':      f'E{i * 3 + j:08d}',
                'dwellings_2011': int(100 + rng.integers(-20, 20)),
                'dwellings_2021': int(110 + rng.integers(-20, 20)),
            }
            for col in ALL_COLS_PLAN:
                row[col] = float(rng.integers(-5, 15))
            for col in ALL_COLS_BEN:
                row[col] = float(rng.integers(-5, 15))
            rows.append(row)

    return gpd.GeoDataFrame(rows, geometry=geometries, crs='EPSG:27700')


@pytest.fixture(scope='session')
def data_dict(synthetic_gdf):
    return make_data_dict(synthetic_gdf)


@pytest.fixture(scope='session')
def mock_trace(data_dict, rng):
    """
    Minimal az.InferenceData suitable for diagnostics tests.
    Contains posterior z, scalar parameters, lambda_weights, alpha_spatial,
    and sample_stats/diverging — all with the right shapes.
    """
    n_areas = data_dict['n_areas']
    n_years = data_dict['n_years']

    z_samples = rng.normal(1.0, 2.0, size=(N_CHAINS, N_DRAWS, n_areas, n_years))

    lw_raw    = rng.random(size=(N_CHAINS * N_DRAWS, 4))
    lw        = (lw_raw / lw_raw.sum(axis=1, keepdims=True)).reshape(N_CHAINS, N_DRAWS, 4)

    return make_idata(
        posterior={
            'z':             z_samples,
            'mu_slab':       rng.normal(1.0, 0.5,   size=(N_CHAINS, N_DRAWS)),
            'sigma_slab':    np.abs(rng.normal(5.0, 1.0, size=(N_CHAINS, N_DRAWS))),
            'lambda_weights': lw,
            'alpha_spatial': np.clip(
                rng.normal(0.05, 0.02, size=(N_CHAINS, N_DRAWS)), 0, 1),
        },
        sample_stats={'diverging': np.zeros((N_CHAINS, N_DRAWS), dtype=bool)},
    )


@pytest.fixture(scope='session')
def mock_trace_with_divergences(data_dict, rng):
    """Like mock_trace but with some divergences for testing divergence detection."""
    n_areas = data_dict['n_areas']
    n_years = data_dict['n_years']

    diverging = np.zeros((N_CHAINS, N_DRAWS), dtype=bool)
    diverging[0, :5] = True  # 5 divergences in chain 0

    return make_idata(
        posterior={
            'z':          rng.normal(0, 1, size=(N_CHAINS, N_DRAWS, n_areas, n_years)),
            'mu_slab':    rng.normal(0, 1, size=(N_CHAINS, N_DRAWS)),
            'sigma_slab': np.abs(rng.normal(5, 1, size=(N_CHAINS, N_DRAWS))),
        },
        sample_stats={'diverging': diverging},
    )


@pytest.fixture(scope='session')
def outlier_gdf(synthetic_gdf):
    """
    Copy of synthetic_gdf with known hard and soft outliers injected.
    Area 0: hard outlier in planning (value > 2000)
    Area 1: hard outlier in BEN (value < -500)
    Area 2: soft outlier (large discrepancy, one source near zero)
    Areas 3-8: clean
    """
    gdf = synthetic_gdf.copy()

    # Hard outlier in planning, area 0, year 0
    gdf.at[0, INFER_COLS_PLAN[0]] = 2500.0

    # Hard outlier in BEN, area 1, year 1
    gdf.at[1, INFER_COLS_BEN[1]] = -600.0

    # Soft outlier: large discrepancy with one source near zero, area 2, year 2
    gdf.at[2, INFER_COLS_PLAN[2]] = 0.0
    gdf.at[2, INFER_COLS_BEN[2]]  = 600.0

    return gdf
