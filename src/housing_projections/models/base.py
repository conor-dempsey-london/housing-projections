from abc import ABC, abstractmethod
import pymc as pm
import arviz as az
import numpy as np
import pytensor.tensor as pt
from pathlib import Path

from housing_projections.config import DEFAULT_SAMPLE_KWARGS, CENSUS_REL_ERROR, CENSUS_ABS_FLOOR


def posterior_predictive_with_zero_snap(model_instance, snap_threshold=0.5):
    """
    Draw posterior predictive samples, snapping near-zero P_like
    values to exactly zero to represent missing planning observations.
    """
    model_instance._require_trace()
    with model_instance.model:
        post_pred = pm.sample_posterior_predictive(model_instance.trace)

    if 'P_like' in post_pred.posterior_predictive:
        P_pred = post_pred.posterior_predictive['P_like'].values
        post_pred.posterior_predictive['P_like'].values[:] = np.where(
            np.abs(P_pred) < snap_threshold, 0.0, P_pred)

    return post_pred


class DwellingModel(ABC):
    """
    Base class for all dwelling inference models.

    Subclasses must implement:
        - build()        — constructs and returns the pm.Model
        - var_names()    — list of scalar parameter names for diagnostics

    Subclasses may override:
        - prior_predictive_checks()
        - posterior_predictive_checks()
        - sample_kwargs                — default sampling configuration
    """

    # ── Default sampling config — override in subclass if needed ─────────────
    # Shared across all models — override in subclass if needed
    nu_obs           = 4.0
    sigma_obs        = 2.0   
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    sample_kwargs    = DEFAULT_SAMPLE_KWARGS

    def __init__(self, data: dict):
        """
        Parameters
        ----------
        data : dict — output of make_data_dict(), contains D, P_obs, E_obs etc.
        """
        self.data    = data
        self.model   = None
        self.trace   = None
        self._built  = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build(self) -> pm.Model:
        """Construct and return the pm.Model. Sets self.model."""
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

    # ── Concrete methods — shared across all models ───────────────────────────

    def __repr__(self):
        return f"{self.name}: {self.description}"

    def sample(self, use_nutpie=True, **kwargs):
        """
        Build (if needed) and sample from the model.
        use_nutpie : bool — use nutpie sampler if available (default True)
        kwargs override self.sample_kwargs.
        """
        if not self._built:
            self.build()
            self._built = True

        merged = {**self.sample_kwargs, **kwargs}

        with self.model:
            if use_nutpie:
                try:
                    import nutpie
                    compiled     = nutpie.compile_pymc_model(self.model)
                    self.trace   = nutpie.sample(
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
                    self.trace = pm.compute_log_likelihood(self.trace)
            else:
                self.trace = pm.sample(**merged)
        return self.trace
    

    def prior_predictive(self, draws=200):
        """Draw prior predictive samples."""
        if not self._built:
            self.build()
            self._built = True

        with self.model:
            return pm.sample_prior_predictive(draws=draws)

    def posterior_predictive(self):
        """Draw posterior predictive samples. Requires trace."""
        self._require_trace()
        with self.model:
            return pm.sample_posterior_predictive(self.trace)
        

    def save(self, results_dir='results/traces'):
        self._require_trace()
        path     = Path(results_dir) / f'{self.name}.nc'
        tmp_path = Path(results_dir) / f'{self.name}_tmp.nc'
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first — always safe since it's a new file
        self.trace.to_netcdf(str(tmp_path))

        # Atomically replace the existing file
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
        
        # Close existing trace if loaded to release file lock
        if self.trace is not None:
            self.trace = None
        
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
        if not self._built:
            self.build()
            self._built = True
        return pm.model_to_graphviz(self.model)

    # ── Shared utilities ──────────────────────────────────────────────────────

    @staticmethod
    def make_sigma_census(D, rel_error=CENSUS_REL_ERROR, abs_floor=CENSUS_ABS_FLOOR):
        return np.maximum(np.abs(D) * rel_error, abs_floor)

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
        except Exception:
            pass