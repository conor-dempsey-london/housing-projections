"""
Tests for housing_projections.models.

Model building is tested here (.build() only — no sampling, which would be
too slow for a unit suite). Sampling behaviour is covered by integration tests.
"""
import numpy as np
import pytest
import pymc as pm

from housing_projections.models.base import DwellingModel
from housing_projections.models.models import (
    # Builder functions
    build_z_prior,
    build_census_constraint,
    build_pre_inference,
    build_lag,
    build_spatial_misallocation,
    # Concrete models
    M0, M0h, M1, M2, M3, M4, M5, M5b, M6,
)
from housing_projections.config import CENSUS_REL_ERROR, CENSUS_ABS_FLOOR


# ── DwellingModel base class ──────────────────────────────────────────────────

class TestMakeSigmaCensus:
    def test_uses_rel_error(self):
        D = np.array([100.0, 200.0, 50.0])
        sigma = DwellingModel.make_sigma_census(D)
        expected_rel = np.abs(D) * CENSUS_REL_ERROR
        np.testing.assert_allclose(
            sigma, np.maximum(expected_rel, CENSUS_ABS_FLOOR))

    def test_floor_applied(self):
        D = np.array([0.0, 1.0])   # rel error is tiny → floor kicks in
        sigma = DwellingModel.make_sigma_census(D)
        assert np.all(sigma >= CENSUS_ABS_FLOOR)

    def test_large_d_uses_rel_error(self):
        D = np.array([10_000.0])
        sigma = DwellingModel.make_sigma_census(D)
        expected = D * CENSUS_REL_ERROR
        assert sigma[0] == pytest.approx(expected[0])

    def test_negative_d_uses_abs(self):
        D = np.array([-200.0])
        sigma = DwellingModel.make_sigma_census(D)
        assert sigma[0] == pytest.approx(200.0 * CENSUS_REL_ERROR)

    def test_custom_params(self):
        D = np.array([100.0])
        sigma = DwellingModel.make_sigma_census(D, rel_error=0.05, abs_floor=1.0)
        assert sigma[0] == pytest.approx(5.0)


class TestNLagsProperty:
    def test_raises_if_max_lag_none(self, data_dict):
        m = M0(data_dict)   # M0 has max_lag=None
        with pytest.raises(AttributeError, match='no lag structure'):
            _ = m.n_lags

    def test_returns_max_lag_plus_one(self, data_dict):
        m = M3(data_dict)   # max_lag=3
        assert m.n_lags == 4

    def test_m4_n_lags(self, data_dict):
        assert M4(data_dict).n_lags == 4


class TestLagAlphaProperty:
    def test_length_equals_n_lags(self, data_dict):
        m = M3(data_dict)
        alpha = m.lag_alpha
        assert len(alpha) == m.n_lags

    def test_decreasing_concentration(self, data_dict):
        """Prior concentrates mass on shorter lags."""
        alpha = M3(data_dict).lag_alpha
        assert alpha[0] >= alpha[1] >= alpha[2]


class TestRequireTrace:
    def test_raises_without_trace(self, data_dict):
        m = M0(data_dict)
        with pytest.raises(RuntimeError, match='No trace found'):
            m._require_trace()


class TestRepr:
    def test_contains_name_and_description(self, data_dict):
        m = M0(data_dict)
        r = repr(m)
        assert m.name in r
        assert m.description in r


# ── Builder functions ─────────────────────────────────────────────────────────

class TestBuildZPrior:
    def test_returns_tuple(self, data_dict):
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        with pm.Model():
            result = build_z_prior(data_dict, n_areas, n_years)
        assert len(result) == 3

    def test_z_shape(self, data_dict):
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        with pm.Model():
            _, _, z = build_z_prior(data_dict, n_areas, n_years)
        assert z.type.shape == (n_areas, n_years)


class TestBuildCensusConstraint:
    def test_adds_census_obs_variable(self, data_dict):
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        D       = data_dict['D']
        sigma_c = DwellingModel.make_sigma_census(D)
        with pm.Model() as model:
            _, _, z = build_z_prior(data_dict, n_areas, n_years)
            build_census_constraint(z, D, sigma_c)
        assert 'census_obs' in model.named_vars


class TestBuildPreInference:
    def test_shape(self, data_dict):
        max_lag = 3
        result  = build_pre_inference(data_dict, max_lag)
        assert result.shape == (data_dict['n_areas'], max_lag)

    def test_dtype_float64(self, data_dict):
        result = build_pre_inference(data_dict, max_lag=3)
        assert result.dtype == np.float64

    def test_uses_planning_data(self, data_dict):
        result = build_pre_inference(data_dict, max_lag=1)
        assert not np.all(result == 0)


class TestBuildLag:
    def test_lambda_weights_created(self, data_dict):
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        n_lags, max_lag  = 4, 3
        alpha = np.array([4.0, 2.0, 1.0, 1.0])
        pre   = build_pre_inference(data_dict, max_lag)

        with pm.Model() as model:
            _, _, z = build_z_prior(data_dict, n_areas, n_years)
            lw, P_mean = build_lag(z, pre, n_areas, n_years, n_lags, alpha, max_lag)
        assert 'lambda_weights' in model.named_vars

    def test_fixed_lambda_weights_not_sampled(self, data_dict):
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        n_lags, max_lag  = 4, 3
        alpha = np.array([4.0, 2.0, 1.0, 1.0])
        pre   = build_pre_inference(data_dict, max_lag)
        fixed = np.array([0.6, 0.2, 0.1, 0.1])

        with pm.Model() as model:
            _, _, z = build_z_prior(data_dict, n_areas, n_years)
            build_lag(z, pre, n_areas, n_years, n_lags, alpha, max_lag,
                      lambda_weights=fixed)
        # lambda_weights should NOT be a free variable when fixed
        assert 'lambda_weights' not in model.named_vars


class TestBuildSpatialMisallocation:
    def test_adds_alpha_spatial(self, data_dict):
        from housing_projections.spatial import build_spatial_weights
        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        W = build_spatial_weights(data_dict['gdf'])
        with pm.Model() as model:
            _, _, z = build_z_prior(data_dict, n_areas, n_years)
            build_spatial_misallocation(z, W, n_areas, n_years)
        assert 'alpha_spatial' in model.named_vars


# ── Concrete model .build() ────────────────────────────────────────────────────

@pytest.mark.parametrize('ModelClass', [M0, M0h, M1, M2, M3, M4, M5, M5b, M6])
class TestModelBuild:
    def test_build_returns_model(self, ModelClass, data_dict):
        m     = ModelClass(data_dict)
        model = m.build()
        assert isinstance(model, pm.Model)

    def test_build_assigns_self_model(self, ModelClass, data_dict):
        m = ModelClass(data_dict)
        m.build()
        assert m.model is not None

    def test_model_has_z_variable(self, ModelClass, data_dict):
        m = ModelClass(data_dict)
        m.build()
        assert 'z' in m.model.named_vars

    def test_model_has_census_obs(self, ModelClass, data_dict):
        m = ModelClass(data_dict)
        m.build()
        assert 'census_obs' in m.model.named_vars

    def test_model_has_e_like(self, ModelClass, data_dict):
        m = ModelClass(data_dict)
        m.build()
        assert 'E_like' in m.model.named_vars

    def test_model_has_p_like(self, ModelClass, data_dict):
        m = ModelClass(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars

    def test_rebuild_replaces_model(self, ModelClass, data_dict):
        m  = ModelClass(data_dict)
        m1 = m.build()
        m2 = m.build()
        assert m.model is m2


# ── Model-specific structure checks ───────────────────────────────────────────

class TestM0Structure:
    def test_var_names(self, data_dict):
        assert set(M0.var_names) == {'mu_slab', 'sigma_slab'}

    def test_snap_zeros_false(self):
        assert M0.snap_zeros is False

    def test_max_lag_none(self):
        assert M0.max_lag is None


class TestM3Structure:
    def test_has_lambda_weights(self, data_dict):
        m = M3(data_dict)
        m.build()
        assert 'lambda_weights' in m.model.named_vars

    def test_max_lag(self):
        assert M3.max_lag == 3

    def test_n_lags(self, data_dict):
        assert M3(data_dict).n_lags == 4


class TestM4Structure:
    def test_has_pi_miss(self, data_dict):
        m = M4(data_dict)
        m.build()
        assert 'pi_miss' in m.model.named_vars

    def test_snap_zeros_true(self):
        assert M4.snap_zeros is True


class TestM5Structure:
    def test_has_asymmetric_pi_miss(self, data_dict):
        m = M5(data_dict)
        m.build()
        assert 'pi_miss_pos' in m.model.named_vars
        assert 'pi_miss_neg' in m.model.named_vars


class TestM5bStructure:
    def test_has_w_tight(self, data_dict):
        m = M5b(data_dict)
        m.build()
        assert 'w_tight' in m.model.named_vars


class TestM6Structure:
    def test_has_alpha_spatial(self, data_dict):
        m = M6(data_dict)
        m.build()
        assert 'alpha_spatial' in m.model.named_vars

    def test_lambda_weights_sampled_by_default(self, data_dict):
        m = M6(data_dict)
        m.build()
        assert 'lambda_weights' in m.model.named_vars

    def test_fixed_lambda_weights_not_sampled(self, data_dict):
        m = M6(data_dict)
        m.lambda_weights_fixed = np.array([0.6, 0.2, 0.1, 0.1])
        m.build()
        assert 'lambda_weights' not in m.model.named_vars

    def test_var_names_excludes_lw_when_fixed(self, data_dict):
        m = M6(data_dict)
        m.lambda_weights_fixed = np.array([0.6, 0.2, 0.1, 0.1])
        assert 'lambda_weights' not in m.var_names

    def test_var_names_includes_lw_when_sampled(self, data_dict):
        m = M6(data_dict)
        assert 'lambda_weights' in m.var_names
