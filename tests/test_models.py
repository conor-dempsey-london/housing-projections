"""Tests for housing_projections.models — .build() only, no sampling."""
import numpy as np
import pymc as pm
import pytest  # noqa: F401

from housing_projections.models.models import M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M0h, M5b


@pytest.mark.parametrize('ModelClass', [M0, M0h, M1, M2, M3, M4, M5, M5b, M6, M7, M9])
class TestModelBuild:
    def test_build_returns_model(self, ModelClass, data_dict):
        assert isinstance(ModelClass(data_dict).build(), pm.Model)

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
        m.build()
        m2 = m.build()
        assert m.model is m2


class TestM0Structure:
    def test_var_names(self):
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
        assert 'lambda_weights' in M6(data_dict).var_names


class TestM7Structure:
    def test_has_rho(self, data_dict):
        m = M7(data_dict)
        m.build()
        assert 'rho' in m.model.named_vars

    def test_has_sigma_innov(self, data_dict):
        m = M7(data_dict)
        m.build()
        assert 'sigma_innov' in m.model.named_vars

    def test_ar1_prior_parameters(self, data_dict):
        m = M7(data_dict)
        m.build()
        assert 'z_init_raw' in m.model.named_vars

    def test_snap_zeros_true(self):
        assert M7.snap_zeros is True


class TestM8Structure:
    def test_raises_without_borough_idx(self, data_dict):
        m = M8(data_dict)
        with pytest.raises(ValueError, match='borough_idx'):
            m.build()

    def test_builds_with_borough_idx(self, data_dict):
        n = data_dict['n_areas']
        data_with_borough = {
            **data_dict,
            'borough_idx': np.zeros(n, dtype=int),
            'n_boroughs': 1,
        }
        m = M8(data_with_borough)
        assert isinstance(m.build(), pm.Model)

    def test_has_mu_borough(self, data_dict):
        n = data_dict['n_areas']
        data_with_borough = {
            **data_dict,
            'borough_idx': np.zeros(n, dtype=int),
            'n_boroughs': 1,
        }
        m = M8(data_with_borough)
        m.build()
        assert 'mu_borough' in m.model.named_vars


class TestM9Structure:
    def test_has_sigma_base_plan(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'sigma_base_plan' in m.model.named_vars

    def test_has_sigma_obs_plan(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'sigma_obs_plan' in m.model.named_vars

    def test_sigma_year_offset_shape(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'sigma_year_offset' in m.model.named_vars
