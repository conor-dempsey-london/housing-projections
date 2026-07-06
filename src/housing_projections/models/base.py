from abc import ABC, abstractmethod

__all__ = ["DwellingModel"]
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from housing_projections.config import (
    CENSUS_ABS_FLOOR,
    CENSUS_REL_ERROR,
    DEFAULT_SAMPLE_KWARGS,
    INFER_YEARS,
)


class DwellingModel(ABC):
    """
    Base class for all dwelling inference models.

    Subclasses must implement:
        - build()        — constructs and returns the pm.Model
        - var_names      — list of scalar parameter names for diagnostics
        - name           — short model identifier e.g. 'M3'
        - description    — one-line description of what the model adds

    Subclasses may override:
        - snap_zeros     — set True to snap near-zero P_like to 0 in
                           posterior predictive (default False)
        - max_lag        — set to an int to unlock n_lags / lag_alpha
                           properties (default None)
        - sample_kwargs  — default sampling configuration
    """

    # ── Shared observation model parameters ──────────────────────────────────
    nu_obs           = 4.0
    sigma_obs        = 2.0
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    # ── Sampling defaults ─────────────────────────────────────────────────────
    sample_kwargs    = DEFAULT_SAMPLE_KWARGS

    # ── Lag structure — override max_lag in subclasses that have a lag ────────
    max_lag: int | None = None

    # ── Posterior predictive zero-snapping ────────────────────────────────────
    # Set True in models with zero-inflated planning likelihood (M4+)
    snap_zeros       = False
    _snap_threshold  = 0.5

    def __init__(self, data: dict):
        """
        Parameters
        ----------
        data : dict — output of make_data_dict(), contains D, P_obs, E_obs etc.
        """
        self.data  = data
        self.model = None
        self.trace = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build(self) -> pm.Model:
        """Construct the pm.Model, assign to self.model, and return it."""
        ...

    @property
    @abstractmethod
    def var_names(self) -> list[str]:
        """Scalar parameter names to include in diagnostics and trace plots."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model name e.g. 'M0', 'M1'. Used for saving traces."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this model adds over the previous."""
        ...

    # ── Derived lag properties ────────────────────────────────────────────────

    @property
    def n_lags(self) -> int:
        """Number of lag terms (max_lag + 1). Requires max_lag to be set."""
        if self.max_lag is None:
            raise AttributeError(
                f"{self.name} has no lag structure (max_lag is None)"
            )
        return self.max_lag + 1

    @property
    def lag_alpha(self) -> np.ndarray:
        """
        Dirichlet concentration prior for lag weights.
        Concentrates mass on shorter lags: [4, 2, 1, 1][:n_lags].
        """
        return np.array([4.0, 2.0, 1.0, 1.0])[:self.n_lags]

    # ── Concrete methods — shared across all models ───────────────────────────

    def __repr__(self):
        return f"{self.name}: {self.description}"

    def sample(self, use_nutpie=True, **kwargs):
        """
        Build (if needed) and sample from the model.

        Parameters
        ----------
        use_nutpie : bool — use nutpie sampler if available (default True)
        **kwargs   : override self.sample_kwargs
        """
        if self.model is None:
            self.build()

        merged = {**self.sample_kwargs, **kwargs}

        with self.model:
            if use_nutpie:
                try:
                    import nutpie
                    compiled   = nutpie.compile_pymc_model(self.model)
                    self.trace = nutpie.sample(
                        compiled,
                        draws         = merged.get('draws',         500),
                        tune          = merged.get('tune',          500),
                        chains        = merged.get('chains',        2),
                        target_accept = merged.get('target_accept', 0.9),
                        seed          = merged.get('random_seed',   42),
                    )
                except ImportError:
                    print("nutpie not installed, falling back to PyMC sampler")
                    self.trace = pm.sample(**merged)
            else:
                self.trace = pm.sample(**merged)
            self.trace = pm.compute_log_likelihood(self.trace)
        return self.trace

    def run(self, results_dir='results/traces', **kwargs):
        """
        Sample, print diagnostics, and save in one call.

        Parameters
        ----------
        results_dir : str
        **kwargs    : passed through to sample()

        Returns
        -------
        az.InferenceData
        """
        self.sample(**kwargs)
        self.diagnostics()
        self.save(results_dir=results_dir)
        return self.trace

    def prior_predictive(self, draws=200):
        """Draw prior predictive samples."""
        if self.model is None:
            self.build()
        with self.model:
            return pm.sample_prior_predictive(draws=draws)

    def posterior_predictive(self):
        """
        Draw posterior predictive samples. Requires trace.

        If snap_zeros is True (set on M4+), near-zero P_like values are
        snapped to exactly 0 to represent missing planning observations.
        """
        self._require_trace()
        with self.model:
            post_pred = pm.sample_posterior_predictive(self.trace)

        if self.snap_zeros and 'P_like' in post_pred.posterior_predictive:
            P_pred = post_pred.posterior_predictive['P_like'].values
            post_pred.posterior_predictive['P_like'].values[:] = np.where(
                np.abs(P_pred) < self._snap_threshold, 0.0, P_pred)

        return post_pred

    def save(self, results_dir='results/traces'):
        self._require_trace()
        path     = Path(results_dir) / f'{self.name}.nc'
        tmp_path = Path(results_dir) / f'{self.name}_tmp.nc'
        path.parent.mkdir(parents=True, exist_ok=True)

        self.trace.to_netcdf(str(tmp_path))

        try:
            tmp_path.replace(path)
        except PermissionError:
            # File is locked by lazy-loaded xarray — close handles and retry once
            print(f"  {self.name}.nc is locked, releasing handles and retrying...")
            self._close_trace()
            tmp_path.replace(path)

        print(f"Saved {self.name} trace to {path}")

    def load(self, results_dir='results/traces'):
        """Load trace from netcdf."""
        path = Path(results_dir) / f'{self.name}.nc'
        self.trace = None   # release any existing file lock
        self.trace = az.from_netcdf(str(path))
        return self.trace

    def diagnostics(self):
        """Print sampling diagnostics for scalar parameters."""
        self._require_trace()
        summary = az.summary(self.trace, var_names=self.var_names)
        print(f"\n── {self.name} diagnostics ──────────────────")
        print(f"   {self.description}")
        print(summary)
        n_divergences = int(self.trace.sample_stats.diverging.sum())
        print(f"   Divergences: {n_divergences}")
        if n_divergences > 0:
            print(f"   *** WARNING: {n_divergences} divergences detected ***")
        return summary

    def graph(self):
        """Display the model graph."""
        if self.model is None:
            self.build()
        return pm.model_to_graphviz(self.model)

    # ── Shared utilities ──────────────────────────────────────────────────────

    @staticmethod
    def make_sigma_census(D, rel_error=CENSUS_REL_ERROR, abs_floor=CENSUS_ABS_FLOOR):
        return np.maximum(np.abs(D) * rel_error, abs_floor)

    def _default_coords(self) -> dict:
        """Standard PyMC model coordinates — embeds LSOA codes and years in the trace."""
        return {
            'area': self.data['gdf']['LSOA21CD'].tolist(),
            'year': INFER_YEARS,
        }

    @staticmethod
    def make_mixture_weights(pi, n_areas, n_years):
        return pt.stack([
            pt.ones((n_areas, n_years)) * pi,
            pt.ones((n_areas, n_years)) * (1 - pi),
        ], axis=-1)

    def add_observation_likelihoods(self, z, P_obs, E_obs,
                                     sigma_plan=None, sigma_ben=None):
        """
        Shared observation model with fixed or provided sigma.
        If sigmas not provided, uses self.sigma_obs for both.
        """
        sigma_plan = sigma_plan if sigma_plan is not None else self.sigma_obs
        sigma_ben  = sigma_ben  if sigma_ben  is not None else self.sigma_obs

        pm.StudentT('P_like', nu=self.nu_obs, mu=z,
                    sigma=sigma_plan, observed=P_obs)
        pm.StudentT('E_like', nu=self.nu_obs, mu=z,
                    sigma=sigma_ben,  observed=E_obs)

    def add_ben_likelihood(self, z, E_obs, sigma_ben=None):
        """
        Add BEN likelihood only — for models where planning has a custom
        likelihood (M4+) and can't use add_observation_likelihoods.
        """
        sigma = sigma_ben if sigma_ben is not None else self.sigma_obs
        pm.StudentT('E_like', nu=self.nu_obs, mu=z,
                    sigma=sigma, observed=E_obs)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_trace(self):
        if self.trace is None:
            raise RuntimeError(
                f"No trace found for {self.name}. "
                f"Run .sample() or .load() first."
            )

    def _close_trace(self):
        """
        Explicitly close all xarray file handles held by the trace.
        Required on Windows before overwriting a netCDF file.
        """
        if self.trace is None:
            return
        try:
            self.trace.close()
        except Exception:  # noqa: BLE001 — intentionally swallow close() errors on Windows
            pass
