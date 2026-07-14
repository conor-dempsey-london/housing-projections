"""Tests for housing_projections.models — .build() only, no sampling."""
import numpy as np
import pymc as pm
import pytest  # noqa: F401

from housing_projections.models.models import (
    AZ0,
    AZ2,
    AZ3,
    AZ4,
    AZ5,
    M0,
    M1,
    M5,
    M6,
    M7,
    M8,
    M9,
    M10,
    M11,
    M12,
    M13,
    M14,
    M15,
    M16,
    AZ0a,
    AZ0b,
    AZ1a,
    AZ1b,
    AZ1c,
    AZ1d,
    AZ1e,
    AZ1f,
    AZ1g,
    AZ1h,
    AZ2b,
    AZ4b,
    M0h,
    M1h,
    _build_fixed_lag,
    _build_hierarchical_lag_pinned,
    _build_pre_inference,
    _build_zero_sum_z_prior,
)


@pytest.mark.parametrize('ModelClass', [M0, M0h, M1, M1h, M5, M6, M8, M9])
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
        assert set(M0.var_names) == {'mu_slab', 'sigma_slab', 'sigma_plan', 'sigma_ben'}

    def test_snap_zeros_false(self):
        assert M0.snap_zeros is False

    def test_max_lag_none(self):
        assert M0.max_lag is None


class TestM1Structure:
    def test_has_lambda_weights(self, data_dict):
        m = M1(data_dict)
        m.build()
        assert 'lambda_weights' in m.model.named_vars

    def test_max_lag(self):
        assert M1.max_lag == 3

    def test_n_lags(self, data_dict):
        assert M1(data_dict).n_lags == 4


class TestM1hStructure:
    def test_has_lambda_weights(self, data_dict):
        m = M1h(data_dict)
        m.build()
        assert 'lambda_weights' in m.model.named_vars

    def test_has_sigma_slab(self, data_dict):
        m = M1h(data_dict)
        m.build()
        assert 'sigma_slab' in m.model.named_vars

    def test_var_names(self):
        assert set(M1h.var_names) == {'sigma_slab', 'sigma_plan', 'sigma_ben', 'lambda_weights'}

    def test_max_lag(self):
        assert M1h.max_lag == 3


class TestM5Structure:
    def test_has_alpha_spatial(self, data_dict):
        m = M5(data_dict)
        m.build()
        assert 'alpha_spatial' in m.model.named_vars

    def test_lambda_weights_sampled_by_default(self, data_dict):
        m = M5(data_dict)
        m.build()
        assert 'lambda_weights' in m.model.named_vars

    def test_fixed_lambda_weights_not_sampled(self, data_dict):
        m = M5(data_dict)
        m.lambda_weights_fixed = np.array([0.6, 0.2, 0.1, 0.1])
        m.build()
        assert 'lambda_weights' not in m.model.named_vars

    def test_var_names_excludes_lw_when_fixed(self, data_dict):
        m = M5(data_dict)
        m.lambda_weights_fixed = np.array([0.6, 0.2, 0.1, 0.1])
        assert 'lambda_weights' not in m.var_names

    def test_var_names_includes_lw_when_sampled(self, data_dict):
        assert 'lambda_weights' in M5(data_dict).var_names


class TestM6Structure:
    def test_has_rho(self, data_dict):
        m = M6(data_dict)
        m.build()
        assert 'rho' in m.model.named_vars

    def test_has_sigma_innov(self, data_dict):
        m = M6(data_dict)
        m.build()
        assert 'sigma_innov' in m.model.named_vars

    def test_ar1_prior_parameters(self, data_dict):
        m = M6(data_dict)
        m.build()
        assert 'z_init_raw' in m.model.named_vars

    def test_snap_zeros_false(self):
        # M6 no longer has a zero-inflated planning likelihood (rebuilt on
        # M1h's plain StudentT likelihood), so no zero-snapping is needed.
        assert M6.snap_zeros is False


class TestM7Structure:
    def test_raises_without_borough_idx(self, data_dict):
        m = M7(data_dict)
        with pytest.raises(ValueError, match='borough_idx'):
            m.build()

    def test_builds_with_borough_idx(self, data_dict):
        n = data_dict['n_areas']
        data_with_borough = {
            **data_dict,
            'borough_idx': np.zeros(n, dtype=int),
            'n_boroughs': 1,
        }
        m = M7(data_with_borough)
        assert isinstance(m.build(), pm.Model)

    def test_has_mu_borough(self, data_dict):
        n = data_dict['n_areas']
        data_with_borough = {
            **data_dict,
            'borough_idx': np.zeros(n, dtype=int),
            'n_boroughs': 1,
        }
        m = M7(data_with_borough)
        m.build()
        assert 'mu_borough' in m.model.named_vars


class TestM8Structure:
    def test_has_sigma_base_plan(self, data_dict):
        m = M8(data_dict)
        m.build()
        assert 'sigma_base_plan' in m.model.named_vars

    def test_has_sigma_obs_plan(self, data_dict):
        m = M8(data_dict)
        m.build()
        assert 'sigma_obs_plan' in m.model.named_vars

    def test_sigma_year_offset_shape(self, data_dict):
        m = M8(data_dict)
        m.build()
        assert 'sigma_year_offset' in m.model.named_vars


class TestM9Structure:
    def test_has_lambda_weights_p(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'lambda_weights_P' in m.model.named_vars

    def test_has_lambda_weights_e(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'lambda_weights_E' in m.model.named_vars

    def test_no_unsuffixed_lambda_weights(self, data_dict):
        # Regression test: _build_lag's default name must not collide when
        # called twice in the same model for P and E.
        m = M9(data_dict)
        m.build()
        assert 'lambda_weights' not in m.model.named_vars

    def test_has_per_area_sigma_slab(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'sigma_slab' in m.model.named_vars
        sigma_slab_rv = m.model.named_vars['sigma_slab']
        assert sigma_slab_rv.eval().shape == (data_dict['n_areas'],)

    def test_has_hyperparameters(self, data_dict):
        m = M9(data_dict)
        m.build()
        assert 'mu_log_sigma' in m.model.named_vars
        assert 'tau_log_sigma' in m.model.named_vars

    def test_var_names(self):
        assert set(M9.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
            'sigma_plan', 'sigma_ben',
            'lambda_weights_P', 'lambda_weights_E',
        }

    def test_max_lag(self):
        assert M9.max_lag == 3

    def test_sample_kwargs_overrides_target_accept(self):
        assert M9.sample_kwargs['target_accept'] == 0.95


class TestM10Structure:
    def test_raises_without_borough_idx(self, data_dict):
        m = M10(data_dict)
        with pytest.raises(ValueError, match='borough_idx'):
            m.build()

    def test_builds_with_borough_idx(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        assert isinstance(m.build(), pm.Model)

    def test_has_lambda_weights_p(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        assert 'lambda_weights_P' in m.model.named_vars

    def test_has_lambda_weights_e(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        assert 'lambda_weights_E' in m.model.named_vars

    def test_has_per_area_sigma_slab(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        assert 'sigma_slab' in m.model.named_vars
        sigma_slab_rv = m.model.named_vars['sigma_slab']
        assert sigma_slab_rv.eval().shape == (data_dict_with_borough['n_areas'],)

    def test_has_per_borough_sigma_slab(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        assert 'sigma_slab_borough' in m.model.named_vars
        sigma_slab_borough_rv = m.model.named_vars['sigma_slab_borough']
        assert sigma_slab_borough_rv.eval().shape == (
            data_dict_with_borough['n_boroughs'],)

    def test_has_kappa_p_and_e(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        for name in ('kappa_P', 'kappa_E'):
            assert name in m.model.named_vars
            assert m.model.named_vars[name].eval().shape == (
                data_dict_with_borough['n_areas'],)

    def test_has_shared_sigma_kappa(self, data_dict_with_borough):
        m = M10(data_dict_with_borough)
        m.build()
        assert 'sigma_kappa' in m.model.named_vars
        # kappa_P/kappa_E built from a single shared sigma_kappa, not two
        assert 'sigma_kappa_P' not in m.model.named_vars
        assert 'sigma_kappa_E' not in m.model.named_vars

    def test_var_names(self):
        assert set(M10.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab_borough',
            'sigma_kappa', 'sigma_plan', 'sigma_ben',
            'lambda_weights_P', 'lambda_weights_E',
        }

    def test_max_lag(self):
        assert M10.max_lag == 3

    def test_sample_kwargs_overrides_target_accept(self):
        assert M10.sample_kwargs['target_accept'] == 0.95

    def test_sigma_kappa_prior_calibrated(self):
        assert M10.sigma_kappa_prior == 0.68


class TestM11Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M11(data_dict).build(), pm.Model)

    def test_has_lambda_weights_p(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'lambda_weights_P' in m.model.named_vars

    def test_has_lambda_weights_e(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'lambda_weights_E' in m.model.named_vars

    def test_no_separate_p_like_or_e_like(self, data_dict):
        # M11 replaces independent P_like/E_like with one joint Potential —
        # regression test that the old per-source likelihood names don't leak in.
        m = M11(data_dict)
        m.build()
        assert 'P_like' not in m.model.named_vars
        assert 'E_like' not in m.model.named_vars

    def test_has_joint_likelihood_potential(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'PE_like' in m.model.named_vars

    def test_has_agreement_prob(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'agreement_prob' in m.model.named_vars
        agreement_prob_rv = m.model.named_vars['agreement_prob']
        assert agreement_prob_rv.eval().shape == (
            data_dict['n_areas'], data_dict['n_years'])

    def test_has_rho_agree(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'rho_agree' in m.model.named_vars

    def test_has_per_area_sigma_slab(self, data_dict):
        m = M11(data_dict)
        m.build()
        assert 'sigma_slab' in m.model.named_vars
        sigma_slab_rv = m.model.named_vars['sigma_slab']
        assert sigma_slab_rv.eval().shape == (data_dict['n_areas'],)

    def test_var_names(self):
        assert set(M11.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_agree', 'lambda_weights_P', 'lambda_weights_E',
        }

    def test_max_lag(self):
        assert M11.max_lag == 3

    def test_sample_kwargs_overrides_target_accept(self):
        assert M11.sample_kwargs['target_accept'] == 0.95

    def test_z_is_deterministic_non_centered(self, data_dict):
        # M11 builds z non-centered (z_raw free RV + Deterministic z),
        # unlike M9/M10's centered pm.Normal('z', ...).
        m = M11(data_dict)
        m.build()
        assert 'z_raw' in m.model.named_vars


class TestM12Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M12(data_dict).build(), pm.Model)

    def test_has_lambda_weights_p(self, data_dict):
        m = M12(data_dict)
        m.build()
        assert 'lambda_weights_P' in m.model.named_vars

    def test_has_lambda_weights_e(self, data_dict):
        m = M12(data_dict)
        m.build()
        assert 'lambda_weights_E' in m.model.named_vars

    def test_has_independent_p_and_e_likelihoods(self, data_dict):
        # M12 replaces M11's single joint 'PE_like' Potential with two
        # independent per-source ones.
        m = M12(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars
        assert 'PE_like' not in m.model.named_vars

    def test_has_independent_agreement_probs(self, data_dict):
        m = M12(data_dict)
        m.build()
        for name in ('agreement_prob_P', 'agreement_prob_E'):
            assert name in m.model.named_vars
            assert m.model.named_vars[name].eval().shape == (
                data_dict['n_areas'], data_dict['n_years'])

    def test_has_independent_rho(self, data_dict):
        m = M12(data_dict)
        m.build()
        assert 'rho_P' in m.model.named_vars
        assert 'rho_E' in m.model.named_vars
        assert 'rho_agree' not in m.model.named_vars

    def test_has_per_area_sigma_slab(self, data_dict):
        m = M12(data_dict)
        m.build()
        assert 'sigma_slab' in m.model.named_vars
        sigma_slab_rv = m.model.named_vars['sigma_slab']
        assert sigma_slab_rv.eval().shape == (data_dict['n_areas'],)

    def test_z_is_deterministic_non_centered(self, data_dict):
        m = M12(data_dict)
        m.build()
        assert 'z_raw' in m.model.named_vars

    def test_var_names(self):
        assert set(M12.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_P', 'rho_E', 'lambda_weights_P', 'lambda_weights_E',
        }

    def test_max_lag(self):
        assert M12.max_lag == 3

    def test_sample_kwargs_overrides_target_accept(self):
        assert M12.sample_kwargs['target_accept'] == 0.95


class TestM13Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M13(data_dict).build(), pm.Model)

    def test_has_independent_p_and_e_likelihoods(self, data_dict):
        m = M13(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars

    def test_has_independent_agreement_probs(self, data_dict):
        m = M13(data_dict)
        m.build()
        for name in ('agreement_prob_P', 'agreement_prob_E'):
            assert name in m.model.named_vars
            assert m.model.named_vars[name].eval().shape == (
                data_dict['n_areas'], data_dict['n_years'])

    def test_has_independent_rho(self, data_dict):
        m = M13(data_dict)
        m.build()
        assert 'rho_P' in m.model.named_vars
        assert 'rho_E' in m.model.named_vars

    def test_has_pi_offset_instead_of_lambda_weights(self, data_dict):
        # M13 removes _build_lag's lambda_weights entirely, replacing it
        # with per-source pi_offset Dirichlets over candidate shifts.
        m = M13(data_dict)
        m.build()
        assert 'pi_offset_P' in m.model.named_vars
        assert 'pi_offset_E' in m.model.named_vars
        assert 'lambda_weights_P' not in m.model.named_vars
        assert 'lambda_weights_E' not in m.model.named_vars
        n_candidates = 2 * M13.max_offset + 1
        assert m.model.named_vars['pi_offset_P'].eval().shape == (n_candidates,)

    def test_has_per_area_sigma_slab(self, data_dict):
        m = M13(data_dict)
        m.build()
        assert 'sigma_slab' in m.model.named_vars
        sigma_slab_rv = m.model.named_vars['sigma_slab']
        assert sigma_slab_rv.eval().shape == (data_dict['n_areas'],)

    def test_z_is_deterministic_non_centered(self, data_dict):
        m = M13(data_dict)
        m.build()
        assert 'z_raw' in m.model.named_vars

    def test_no_lag_machinery(self):
        # M13 has no _build_lag call, so max_lag/n_lags/lag_alpha are unused.
        assert M13.max_lag is None

    def test_var_names(self):
        assert set(M13.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E',
        }

    def test_sample_kwargs_overrides_target_accept(self):
        assert M13.sample_kwargs['target_accept'] == 0.95


class TestM14Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M14(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        # z sums to D exactly by construction (see
        # _build_z_prior_profile_library) -- no soft census likelihood.
        m = M14(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_z_sums_to_census_exactly(self, data_dict):
        m = M14(data_dict)
        m.build()
        prior = m.prior_predictive(draws=20)
        z = prior.prior['z'].values  # (chain, draw, area, year)
        resid = np.abs(z.sum(axis=-1) - data_dict['D'][None, None, :])
        assert resid.max() < 1e-8

    def test_has_profile_k_and_amplitude(self, data_dict):
        m = M14(data_dict)
        m.build()
        assert 'profile_k' in m.model.named_vars
        assert 'amplitude' in m.model.named_vars
        assert 'pi_profile' in m.model.named_vars
        n_years = data_dict['n_years']
        assert m.model.named_vars['pi_profile'].eval().shape == (n_years + 1,)

    def test_profile_k_is_discrete(self, data_dict):
        m = M14(data_dict)
        m.build()
        assert isinstance(m.model.named_vars['profile_k'].owner.op, pm.Categorical)

    def test_has_independent_p_and_e_likelihoods(self, data_dict):
        m = M14(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars

    def test_has_pi_offset(self, data_dict):
        m = M14(data_dict)
        m.build()
        assert 'pi_offset_P' in m.model.named_vars
        assert 'pi_offset_E' in m.model.named_vars

    def test_has_per_area_sigma_slab(self, data_dict):
        m = M14(data_dict)
        m.build()
        assert m.model.named_vars['sigma_slab'].eval().shape == (data_dict['n_areas'],)

    def test_var_names(self):
        assert set(M14.var_names) == {
            'mu_log_sigma', 'tau_log_sigma', 'sigma_slab', 'amplitude', 'pi_profile',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E',
        }

    def test_sample_kwargs_overrides_target_accept(self):
        assert M14.sample_kwargs['target_accept'] == 0.95

    def test_sample_forces_no_nutpie(self, data_dict, monkeypatch):
        # profile_k is discrete -- nutpie can't compile it, so sample()
        # must always fall back to PyMC's own sampler regardless of the
        # use_nutpie argument.
        m = M14(data_dict)
        seen = {}

        def fake_super_sample(self, use_nutpie=True, **kwargs):
            seen['use_nutpie'] = use_nutpie

        monkeypatch.setattr(
            'housing_projections.models.base.DwellingModel.sample', fake_super_sample)
        M14.sample(m, use_nutpie=True)
        assert seen['use_nutpie'] is False


class TestM15Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M15(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = M15(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_z_sums_to_census_exactly(self, data_dict):
        m = M15(data_dict)
        m.build()
        prior = m.prior_predictive(draws=20)
        z = prior.prior['z'].values
        resid = np.abs(z.sum(axis=-1) - data_dict['D'][None, None, :])
        assert resid.max() < 1e-8

    def test_no_null_row_in_profile_library(self, data_dict):
        # M15 drops the null row that M14 found redundant with
        # amplitude-shrinkage -- profile_k ranges over n_years rows, not
        # n_years + 1.
        m = M15(data_dict)
        m.build()
        assert m.model.named_vars['pi_profile'].eval().shape == (data_dict['n_years'],)

    def test_has_horseshoe_components(self, data_dict):
        m = M15(data_dict)
        m.build()
        for name in ('tau_amplitude', 'lam_amplitude', 'c2_amplitude', 'amplitude'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lam_amplitude'].eval().shape == (data_dict['n_areas'],)

    def test_no_sigma_slab_hierarchy(self, data_dict):
        # The mu_log_sigma/tau_log_sigma/sigma_slab hierarchy from M9-M14
        # is replaced entirely by the horseshoe components above.
        m = M15(data_dict)
        m.build()
        assert 'sigma_slab' not in m.model.named_vars
        assert 'mu_log_sigma' not in m.model.named_vars

    def test_profile_k_is_discrete(self, data_dict):
        m = M15(data_dict)
        m.build()
        assert isinstance(m.model.named_vars['profile_k'].owner.op, pm.Categorical)

    def test_has_independent_p_and_e_likelihoods(self, data_dict):
        m = M15(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars

    def test_has_pi_offset(self, data_dict):
        m = M15(data_dict)
        m.build()
        assert 'pi_offset_P' in m.model.named_vars
        assert 'pi_offset_E' in m.model.named_vars

    def test_var_names(self):
        assert set(M15.var_names) == {
            'tau_amplitude', 'c2_amplitude', 'amplitude', 'pi_profile',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E',
        }

    def test_sample_kwargs_overrides_target_accept(self):
        assert M15.sample_kwargs['target_accept'] == 0.95

    def test_sample_forces_no_nutpie(self, data_dict, monkeypatch):
        m = M15(data_dict)
        seen = {}

        def fake_super_sample(self, use_nutpie=True, **kwargs):
            seen['use_nutpie'] = use_nutpie

        monkeypatch.setattr(
            'housing_projections.models.base.DwellingModel.sample', fake_super_sample)
        M15.sample(m, use_nutpie=True)
        assert seen['use_nutpie'] is False


class TestM16Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(M16(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = M16(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_z_not_in_named_vars(self, data_dict):
        # z must NOT be a Deterministic here -- pymc_extras.marginalize()
        # refuses to marginalise profile_k if any Deterministic depends
        # on it (see _build_z_prior_profile_library_horseshoe's docstring).
        m = M16(data_dict)
        m.build()
        assert 'z' not in m.model.named_vars

    def test_has_horseshoe_components(self, data_dict):
        m = M16(data_dict)
        m.build()
        for name in ('tau_amplitude', 'lam_amplitude', 'c2_amplitude', 'amplitude'):
            assert name in m.model.named_vars

    def test_has_customdist_likelihoods(self, data_dict):
        m = M16(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars

    def test_var_names(self):
        assert set(M16.var_names) == {
            'tau_amplitude', 'c2_amplitude', 'amplitude', 'pi_profile',
            'sigma_agree_plan', 'sigma_agree_ben',
            'sigma_disagree_plan', 'sigma_disagree_ben',
            'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E',
        }

    def test_sample_kwargs_overrides_target_accept(self):
        assert M16.sample_kwargs['target_accept'] == 0.95


@pytest.mark.slow
class TestM16Sampling:
    """
    M16's sample() does something none of the other models do (marginalise
    profile_k, sample, recover it, reconstruct z/agreement_prob/pointwise
    log-likelihood in numpy) -- worth its own slow integration test rather
    than folding into TestSamplingPipeline below, since a structural-only
    test can't catch bugs in that reconstruction pipeline.
    """

    def test_full_pipeline_reconstructs_expected_posterior(self, data_dict):
        m = M16(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'profile_k' in m.trace.posterior
        assert 'agreement_prob_P' in m.trace.posterior
        assert 'agreement_prob_E' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        n_areas, n_years = data_dict['n_areas'], data_dict['n_years']
        z = m.trace.posterior['z'].values
        assert z.shape[-2:] == (n_areas, n_years)

        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        assert m.trace.log_likelihood['P_like'].values.shape == z.shape
        assert m.trace.log_likelihood['E_like'].values.shape == z.shape


class TestAZ0Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ0(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        # z sums to D exactly by construction (ZeroSumNormal) — no soft
        # census likelihood needed, unlike M9-M13.
        m = AZ0(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ0(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ0(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_backward_reallocation_components(self, data_dict):
        m = AZ0(data_dict)
        m.build()
        for name in ('rho_P', 'rho_E',
                     'resp_same_P', 'resp_prior_P', 'resp_noise_P',
                     'resp_same_E', 'resp_prior_E', 'resp_noise_E',
                     'P_like', 'E_like'):
            assert name in m.model.named_vars

    def test_rho_is_three_way_dirichlet(self, data_dict):
        m = AZ0(data_dict)
        m.build()
        assert m.model.named_vars['rho_P'].eval().shape == (3,)
        assert m.model.named_vars['rho_E'].eval().shape == (3,)

    def test_var_names(self):
        assert set(AZ0.var_names) == {
            'sigma_obs_P', 'sigma_obs_E', 'sigma_noise_P', 'sigma_noise_E',
            'rho_P', 'rho_E',
        }

    def test_sample_kwargs_overrides_target_accept(self):
        assert AZ0.sample_kwargs['target_accept'] == 0.95


@pytest.mark.slow
class TestAZ0Sampling:
    def test_full_pipeline(self, data_dict):
        m = AZ0(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        for name in ('resp_same_P', 'resp_prior_P', 'resp_noise_P'):
            resp = m.trace.posterior[name].values
            assert resp.min() >= 0.0 and resp.max() <= 1.0

        resp_sum = (m.trace.posterior['resp_same_P'].values
                    + m.trace.posterior['resp_prior_P'].values
                    + m.trace.posterior['resp_noise_P'].values)
        assert np.abs(resp_sum - 1.0).max() < 1e-6


class TestAZ0aStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ0a(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ0a(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ0a(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ0a(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_mixture_components(self, data_dict):
        # The whole point of AZ0a is to strip AZ0's reallocation mixture
        # back out — none of its machinery should be present.
        m = AZ0a(data_dict)
        m.build()
        for name in ('rho_P', 'rho_E', 'resp_same_P', 'resp_prior_P',
                     'resp_noise_P', 'sigma_noise_P', 'sigma_noise_E'):
            assert name not in m.model.named_vars

    def test_has_plain_likelihoods(self, data_dict):
        m = AZ0a(data_dict)
        m.build()
        assert 'P_like' in m.model.named_vars
        assert 'E_like' in m.model.named_vars
        assert 'sigma_plan' in m.model.named_vars
        assert 'sigma_ben' in m.model.named_vars

    def test_var_names(self):
        assert set(AZ0a.var_names) == {'sigma_plan', 'sigma_ben'}


@pytest.mark.slow
class TestAZ0aSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ0a(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ0bStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ0b(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ0b(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ0b(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ0b(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_noise_branch(self, data_dict):
        # The whole point of AZ0b vs AZ0 is dropping the noise branch that
        # broke AZ0's convergence -- none of its machinery should exist.
        m = AZ0b(data_dict)
        m.build()
        for name in ('sigma_noise_P', 'sigma_noise_E',
                     'resp_prior_P', 'resp_noise_P', 'resp_prior_E', 'resp_noise_E'):
            assert name not in m.model.named_vars

    def test_has_2way_mixture_components(self, data_dict):
        m = AZ0b(data_dict)
        m.build()
        for name in ('rho_P', 'rho_E', 'resp_same_P', 'resp_same_E',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['rho_P'].eval().shape == ()

    def test_var_names(self):
        assert set(AZ0b.var_names) == {'sigma_plan', 'sigma_ben', 'rho_P', 'rho_E'}


@pytest.mark.slow
class TestAZ0bSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ0b(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        resp = m.trace.posterior['resp_same_P'].values
        assert resp.min() >= 0.0 and resp.max() <= 1.0


class TestAZ1aStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1a(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ1a(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ1a(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1a(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_discrete_mixture_machinery(self, data_dict):
        # The whole point of AZ1a vs AZ0b is a smooth convolution instead
        # of a discrete same-year/prior-year mixture -- none of AZ0b's
        # mixture-specific variables should exist here.
        m = AZ1a(data_dict)
        m.build()
        for name in ('rho_P', 'rho_E', 'resp_same_P', 'resp_same_E'):
            assert name not in m.model.named_vars

    def test_has_separate_lag_weights_for_p_and_e(self, data_dict):
        m = AZ1a(data_dict)
        m.build()
        for name in ('lambda_weights_P', 'lambda_weights_E',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        n_lags = AZ1a.max_lag + 1
        assert m.model.named_vars['lambda_weights_P'].eval().shape == (n_lags,)
        assert m.model.named_vars['lambda_weights_E'].eval().shape == (n_lags,)

    def test_var_names(self):
        assert set(AZ1a.var_names) == {
            'sigma_plan', 'sigma_ben', 'lambda_weights_P', 'lambda_weights_E'}


@pytest.mark.slow
class TestAZ1aSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1a(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        lambda_P = m.trace.posterior['lambda_weights_P'].values
        assert lambda_P.min() >= 0.0
        np.testing.assert_allclose(lambda_P.sum(axis=-1), 1.0, atol=1e-6)


class TestAZ1bStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1b(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ1b(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ1b(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1b(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_shared_kernel_machinery(self, data_dict):
        # AZ1b replaces AZ1a's single shared Dirichlet kernel entirely --
        # none of that machinery should exist here.
        m = AZ1b(data_dict)
        m.build()
        for name in ('lambda_weights_P', 'lambda_weights_E'):
            assert name not in m.model.named_vars

    def test_has_per_area_hierarchical_lag_weights(self, data_dict):
        m = AZ1b(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_lags = AZ1b.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_tau', 'lag_P_raw_offset',
                     'lag_P_lambda_weights', 'lag_E_mu_logit', 'lag_E_tau',
                     'lag_E_raw_offset', 'lag_E_lambda_weights',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lag_P_lambda_weights'].eval().shape == (n_areas, n_lags)
        assert m.model.named_vars['lag_P_mu_logit'].eval().shape == (n_lags - 1,)

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1b(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_tau_zero_recovers_shared_kernel(self, data_dict):
        # Sanity check on the hierarchical construction itself: if tau is
        # pinned to 0, every area's lambda_weights must collapse to the
        # exact same vector (softmax(mu_logit)) -- this is the "tau -> 0
        # recovers AZ1a's fully-pooled answer" claim from the docstring,
        # checked directly rather than just asserted.
        m = AZ1b(data_dict)
        model = m.build()
        with model:
            fixed = pm.do(model, {'lag_P_tau': np.zeros(AZ1b.max_lag)})
            draw = pm.draw(fixed['lag_P_lambda_weights'], draws=1, random_seed=0)
        # All areas should be identical when tau=0, regardless of
        # raw_offset's value (it's multiplied by tau=0).
        assert np.allclose(draw, draw[0:1], atol=1e-9)

    def test_var_names(self):
        assert set(AZ1b.var_names) == {
            'sigma_plan', 'sigma_ben',
            'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau'}


@pytest.mark.slow
class TestAZ1bSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1b(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        lambda_P = m.trace.posterior['lag_P_lambda_weights'].values
        assert lambda_P.min() >= 0.0
        np.testing.assert_allclose(lambda_P.sum(axis=-1), 1.0, atol=1e-6)


class TestHierarchicalLagPinned:
    """
    Direct builder-level tests for _build_hierarchical_lag_pinned — not
    wired into a registered AZ-family model; built for the AZ1d/AZ4
    ablation (docs/az-ess-diagnosis.md) that pins specific areas' lag
    logit at a supplied value to test whether they, not general funnel
    geometry, are what drags down population-level lag scalars.
    """

    def _build(self, data_dict, pinned_mask, pinned_logit):
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        max_lag = 2
        n_lags  = max_lag + 1
        prior_logit = np.log(np.array([2.0, 1.0]) / 4.0)
        pre_inference_P = _build_pre_inference(data_dict, max_lag, source='P')
        coords = {'area': list(range(n_areas)), 'year': list(range(n_years))}

        with pm.Model(coords=coords) as model:
            _, _, _, z = _build_zero_sum_z_prior(
                data_dict['D'], n_areas, n_years, floor=3.0, k=0.08)
            _build_hierarchical_lag_pinned(
                z, pre_inference_P, n_areas, n_years, n_lags, max_lag,
                prior_logit, pinned_mask, pinned_logit, name='lag_P')
        return model

    def test_build_returns_model(self, data_dict):
        n_areas = data_dict['n_areas']
        mask = np.zeros(n_areas, dtype=bool)
        logit = np.zeros((n_areas, 2))
        assert isinstance(self._build(data_dict, mask, logit), pm.Model)

    def test_shapes(self, data_dict):
        n_areas = data_dict['n_areas']
        mask = np.zeros(n_areas, dtype=bool)
        logit = np.zeros((n_areas, 2))
        model = self._build(data_dict, mask, logit)
        assert model.named_vars['lag_P_lambda_weights'].eval().shape == (n_areas, 3)
        assert model.named_vars['lag_P_mu_logit'].eval().shape == (2,)
        assert model.named_vars['lag_P_raw_offset'].eval().shape == (n_areas, 2)

    def test_lambda_weights_are_simplices(self, data_dict):
        n_areas = data_dict['n_areas']
        mask = np.array([True, False] * (n_areas // 2) + [False] * (n_areas % 2))
        logit = np.tile([1.5, -0.5], (n_areas, 1))
        model = self._build(data_dict, mask, logit)
        with model:
            draws = pm.draw(model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_pinned_areas_ignore_raw_offset_and_tau(self, data_dict):
        # A pinned area's lambda_weights must depend ONLY on pinned_logit --
        # changing raw_offset or tau (via pm.do) must leave it exactly fixed
        # at softmax([0, *pinned_logit]), the same "mechanically decoupled"
        # check AZ1b's own test_tau_zero_recovers_shared_kernel uses for its
        # tau->0 claim.
        n_areas = data_dict['n_areas']
        mask = np.zeros(n_areas, dtype=bool)
        mask[0] = True
        pinned_row = np.array([2.0, -1.0])
        logit = np.zeros((n_areas, 2))
        logit[0] = pinned_row
        model = self._build(data_dict, mask, logit)

        expected = np.exp(np.concatenate([[0.0], pinned_row]))
        expected = expected / expected.sum()

        with model:
            fixed = pm.do(model, {
                'lag_P_raw_offset': np.full((n_areas, 2), 7.0),
                'lag_P_tau':        np.array([3.0, 3.0]),
            })
            draw = pm.draw(fixed['lag_P_lambda_weights'], draws=3, random_seed=0)
        np.testing.assert_allclose(
            draw[:, 0, :], np.broadcast_to(expected, (3, 3)), atol=1e-6)

    def test_unpinned_areas_match_unpinned_builder(self, data_dict):
        # With every area unpinned, this must reduce EXACTLY to
        # _build_hierarchical_lag's own formula (mu_logit + raw_offset*tau),
        # evaluated at the same fixed values -- confirms the mask/blend
        # doesn't perturb the non-pinned path at all.
        n_areas = data_dict['n_areas']
        mask = np.zeros(n_areas, dtype=bool)
        logit = np.zeros((n_areas, 2))
        model = self._build(data_dict, mask, logit)

        raw_offset_val = np.linspace(-1, 1, n_areas * 2).reshape(n_areas, 2)
        mu_logit_val = np.array([0.3, -0.2])
        tau_val = np.array([1.2, 0.7])

        with model:
            fixed = pm.do(model, {
                'lag_P_raw_offset': raw_offset_val,
                'lag_P_mu_logit':   mu_logit_val,
                'lag_P_tau':        tau_val,
            })
            draw = pm.draw(fixed['lag_P_lambda_weights'], draws=1, random_seed=0)
        if draw.shape != (n_areas, 3):
            draw = draw[0]

        area_logit = mu_logit_val[None, :] + raw_offset_val * tau_val[None, :]
        full_logit = np.concatenate([np.zeros((n_areas, 1)), area_logit], axis=1)
        expected = np.exp(full_logit)
        expected = expected / expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(draw, expected, atol=1e-6)


class TestFixedLag:
    """
    Direct builder-level tests for _build_fixed_lag -- a zero-sampled-
    parameter lag convolution, built for the AZ4 ablation's third arm
    (docs/az-ess-diagnosis.md) that removes ALL of E's lag-mechanism
    sampling uncertainty (not just specific areas' data).
    """

    def test_shared_across_areas_and_matches_manual_softmax(self, data_dict):
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        max_lag = 2
        n_lags  = max_lag + 1
        fixed_logit = np.array([1.5, -0.5])
        pre_inference = _build_pre_inference(data_dict, max_lag, source='E')

        with pm.Model():
            z = pm.Normal('z', 0, 1, shape=(n_areas, n_years))
            lambda_weights, mean = _build_fixed_lag(
                z, pre_inference, n_areas, n_years, n_lags, max_lag,
                fixed_logit, name='lag_E')

        weights = lambda_weights.eval()
        expected = np.exp(np.concatenate([[0.0], fixed_logit]))
        expected = expected / expected.sum()
        assert weights.shape == (n_lags,)
        np.testing.assert_allclose(weights, expected, atol=1e-6)
        assert mean.eval().shape == (n_areas, n_years)

    def test_no_free_random_variables(self, data_dict):
        # The whole point: no mu_logit/tau/raw_offset RVs at all.
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        max_lag = 2
        pre_inference = _build_pre_inference(data_dict, max_lag, source='E')

        with pm.Model() as model:
            z = pm.Normal('z', 0, 1, shape=(n_areas, n_years))
            _build_fixed_lag(z, pre_inference, n_areas, n_years, max_lag + 1,
                             max_lag, np.array([1.5, -0.5]), name='lag_E')
        assert len(model.free_RVs) == 1  # just z


class TestAZ1cStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1c(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1c(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_capped_tau_machinery(self, data_dict):
        m = AZ1c(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_lags = AZ1c.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_tau_frac', 'lag_P_tau',
                     'lag_P_raw_offset', 'lag_P_lambda_weights',
                     'lag_E_mu_logit', 'lag_E_tau_frac', 'lag_E_tau',
                     'lag_E_raw_offset', 'lag_E_lambda_weights',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lag_P_lambda_weights'].eval().shape == (n_areas, n_lags)

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1c(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_tau_never_exceeds_cap(self, data_dict):
        # The whole point of AZ1c over AZ1b: tau is a HARD ceiling
        # (tau_cap * Beta(2,2)), not just a smaller HalfNormal scale --
        # verify it can never exceed tau_cap regardless of draw.
        m = AZ1c(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['lag_P_tau'], m.model['lag_E_tau']],
                draws=200, random_seed=0)
        for d in draws:
            assert d.min() >= 0.0
            assert d.max() <= AZ1c.tau_cap

    def test_tau_zero_recovers_shared_kernel(self, data_dict):
        m = AZ1c(data_dict)
        model = m.build()
        with model:
            fixed = pm.do(model, {'lag_P_tau': np.zeros(AZ1c.max_lag)})
            draw = pm.draw(fixed['lag_P_lambda_weights'], draws=1, random_seed=0)
        assert np.allclose(draw, draw[0:1], atol=1e-9)

    def test_var_names(self):
        assert set(AZ1c.var_names) == {
            'sigma_plan', 'sigma_ben',
            'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau'}


@pytest.mark.slow
class TestAZ1cSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1c(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        tau_P = m.trace.posterior['lag_P_tau'].values
        assert tau_P.max() <= AZ1c.tau_cap


class TestAZ1dStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1d(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1d(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_e_lag_machinery(self, data_dict):
        # The whole point of AZ1d: E's hierarchical lag machinery must be
        # entirely absent -- only P's remains.
        m = AZ1d(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_raw_offset',
                     'lag_E_lambda_weights'):
            assert name not in m.model.named_vars
        for name in ('lag_P_mu_logit', 'lag_P_tau', 'lag_P_raw_offset',
                     'lag_P_lambda_weights', 'sigma_plan', 'sigma_ben',
                     'P_like', 'E_like'):
            assert name in m.model.named_vars

    def test_var_names(self):
        assert set(AZ1d.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_tau'}


@pytest.mark.slow
class TestAZ1dSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1d(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        assert 'lag_E_lambda_weights' not in m.trace.posterior


class TestAZ1eStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1e(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1e(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_horseshoe_machinery(self, data_dict):
        m = AZ1e(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_lags = AZ1e.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_local_scale',
                     'lag_P_tau', 'lag_P_raw_offset', 'lag_P_lambda_weights',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lag_P_local_scale'].eval().shape == (n_areas, n_lags - 1)
        assert m.model.named_vars['lag_P_global_tau'].eval().shape == (n_lags - 1,)

    def test_no_e_lag_machinery(self, data_dict):
        m = AZ1e(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights'):
            assert name not in m.model.named_vars

    def test_tau_varies_per_area(self, data_dict):
        # The whole point of AZ1e over AZ1d: tau is per-area, not one
        # shared scalar -- verify it actually differs across areas for a
        # given draw (not collapsed to a constant).
        m = AZ1e(data_dict)
        m.build()
        with m.model:
            tau = pm.draw(m.model['lag_P_tau'], draws=1, random_seed=0)
        assert tau.shape[0] == data_dict['n_areas']
        assert tau.std(axis=0).min() > 0.0

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1e(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_var_names(self):
        assert set(AZ1e.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau'}


@pytest.mark.slow
class TestAZ1eSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1e(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ1fStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1f(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1f(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_marginalized_machinery(self, data_dict):
        m = AZ1f(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        n_lags = AZ1f.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_tau', 'lag_P_raw_offset',
                     'lag_P_lambda_weights', 'sigma_plan', 'sigma_ben',
                     'P_like', 'P_like_pointwise', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['P_like_pointwise'].eval().shape == (n_areas, n_years)
        assert m.model.named_vars['lag_P_lambda_weights'].eval().shape == (n_areas, n_lags)

    def test_no_p_mean_blend(self, data_dict):
        # The whole point of AZ1f over AZ1d: no single blended P_mean is
        # ever formed -- the likelihood marginalizes over per-category
        # means directly instead.
        m = AZ1f(data_dict)
        m.build()
        assert 'P_mean' not in m.model.named_vars

    def test_no_e_lag_machinery(self, data_dict):
        m = AZ1f(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights'):
            assert name not in m.model.named_vars

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1f(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_var_names(self):
        assert set(AZ1f.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_tau'}


@pytest.mark.slow
class TestAZ1fSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1f(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ1gStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1g(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1g(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_regularized_horseshoe_machinery(self, data_dict):
        m = AZ1g(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_lags = AZ1g.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_local_lambda',
                     'lag_P_tau', 'lag_P_raw_offset', 'lag_P_lambda_weights',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lag_P_local_lambda'].eval().shape == (n_areas, n_lags - 1)
        assert m.model.named_vars['lag_P_global_tau'].eval().shape == (n_lags - 1,)

    def test_no_e_lag_machinery(self, data_dict):
        m = AZ1g(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights'):
            assert name not in m.model.named_vars

    def test_tau_is_bounded_by_slab_regardless_of_local_lambda(self, data_dict):
        # The whole point of AZ1g over AZ1e: even if local_lambda draws an
        # extreme value (the exact mechanism that produced AZ1e's 3-million-
        # fold local_scale blowup and divergences), tau must stay bounded
        # near slab_scale, not explode alongside it. Evaluate the tau
        # expression directly at an extreme local_lambda via symbolic
        # substitution, rather than relying on the model's own RNG to ever
        # draw something that extreme.
        import pytensor

        m = AZ1g(data_dict)
        m.build()
        global_tau_var = m.model['lag_P_global_tau']
        local_lambda_var = m.model['lag_P_local_lambda']
        tau_var = m.model['lag_P_tau']

        global_tau_val = np.full(global_tau_var.eval().shape, 3.0)
        extreme_lambda = np.full(local_lambda_var.eval().shape, 1e6)
        f = pytensor.function([local_lambda_var, global_tau_var], tau_var,
                              on_unused_input='ignore')
        tau_at_extreme = f(extreme_lambda, global_tau_val)
        assert np.all(tau_at_extreme < AZ1g.slab_scale * 1.5)

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1g(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_var_names(self):
        assert set(AZ1g.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau'}


@pytest.mark.slow
class TestAZ1gSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1g(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ1hStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ1h(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ1h(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_canonical_horseshoe_machinery(self, data_dict):
        m = AZ1h(data_dict)
        m.build()
        n_areas = data_dict['n_areas']
        n_lags = AZ1h.max_lag + 1
        for name in ('lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_local_lambda',
                     'lag_P_c2', 'lag_P_tau', 'lag_P_raw_offset', 'lag_P_lambda_weights',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['lag_P_local_lambda'].eval().shape == (n_areas, n_lags - 1)
        assert m.model.named_vars['lag_P_c2'].eval().shape == (n_lags - 1,)
        assert m.model.named_vars['lag_P_global_tau'].eval().shape == (n_lags - 1,)

    def test_no_e_lag_machinery(self, data_dict):
        m = AZ1h(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights'):
            assert name not in m.model.named_vars

    def test_p0_clamped_on_small_datasets(self, data_dict):
        # data_dict's synthetic fixture has far fewer areas than AZ1h.p0=15 -- the
        # tau0 = p0/(n_areas-p0)/sqrt(n_years) formula would divide by a negative
        # number without the clamp in _build_hierarchical_lag_regularized_horseshoe_v2.
        # Just confirming the model builds and global_tau's prior is well-defined
        # (finite, positive beta) rather than erroring or producing NaN/inf.
        m = AZ1h(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_global_tau'], draws=5, random_seed=0)
        assert np.all(np.isfinite(draws))
        assert np.all(draws > 0)

    def test_slab_c2_is_sampled_not_fixed(self, data_dict):
        # The whole point of AZ1h over AZ1g: c2 (the slab) is a genuine random
        # variable with its own prior, not a hand-fixed constant -- confirm draws
        # actually vary rather than being pinned to one value.
        m = AZ1h(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_c2'], draws=20, random_seed=0)
        assert draws.std(axis=0).min() > 0.0

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ1h(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_var_names(self):
        assert set(AZ1h.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_c2'}


@pytest.mark.slow
class TestAZ1hSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ1h(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ2Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ2(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ2(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ2(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ2(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_no_fixed_floor_k_formula(self, data_dict):
        # AZ2 layers a top-boost term on top of the fixed floor + k*|D|
        # base -- sigma_delta must be a Deterministic (a function of a
        # sampled RV, sigma_delta_top_boost), not the fixed numpy array
        # AZ0a/AZ1a/AZ1b use directly.
        m = AZ2(data_dict)
        m.build()
        assert 'sigma_delta' in m.model.named_vars
        assert 'sigma_delta' in [v.name for v in m.model.deterministics]

    def test_no_banded_hierarchy_machinery(self, data_dict):
        # Regression test: AZ2 moved from a 4-band hierarchy (shown on
        # real data to over-complicate this -- see
        # docs/az-family-work-plan.md) to a single top-tier boost. None of
        # the abandoned hierarchy's machinery should exist here.
        m = AZ2(data_dict)
        m.build()
        for name in ('mu_log_sigma_delta', 'tau_log_sigma_delta', 'sigma_delta_band'):
            assert name not in m.model.named_vars

    def test_has_top_boost(self, data_dict):
        m = AZ2(data_dict)
        m.build()
        for name in ('sigma_delta_top_boost', 'sigma_delta',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['sigma_delta_top_boost'].eval().shape == ()
        n_areas = data_dict['n_areas']
        assert m.model.named_vars['sigma_delta'].eval().shape == (n_areas,)

    def test_top_boost_is_positive(self, data_dict):
        m = AZ2(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['sigma_delta_top_boost'], draws=20, random_seed=0)
        assert draws.min() > 0.0

    def test_sigma_delta_never_below_floor(self, data_dict):
        # Regression test from the abandoned 4-band version: sigma_delta
        # must never drop below the fixed floor for any area.
        m = AZ2(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['sigma_delta'], draws=20, random_seed=0)
        assert draws.min() >= AZ2.sigma_delta_floor

    def test_only_top_quartile_gets_boost(self, data_dict):
        # sigma_delta for non-top areas should equal exactly floor +
        # k*|D| (no boost applied); top-quartile areas should exceed that
        # by exactly the sampled top_boost value.
        m = AZ2(data_dict)
        m.build()
        D = data_dict['D']
        top_cutoff = np.quantile(np.abs(D), AZ2.top_quantile)
        is_top = np.abs(D) >= top_cutoff
        with m.model:
            # Draw both from ONE call so they share the same underlying
            # random draw of top_boost (separate pm.draw calls, even with
            # the same seed, aren't guaranteed to agree).
            sigma_delta, top_boost = pm.draw(
                [m.model['sigma_delta'], m.model['sigma_delta_top_boost']],
                draws=1, random_seed=0)
        base = AZ2.sigma_delta_floor + AZ2.k_sigma_delta * np.abs(D)
        np.testing.assert_allclose(sigma_delta[~is_top], base[~is_top], atol=1e-6)
        np.testing.assert_allclose(sigma_delta[is_top], base[is_top] + top_boost, atol=1e-6)

    def test_var_names(self):
        assert set(AZ2.var_names) == {
            'sigma_plan', 'sigma_ben', 'sigma_delta_top_boost'}


@pytest.mark.slow
class TestAZ2Sampling:
    def test_full_pipeline(self, data_dict):
        m = AZ2(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        top_boost = m.trace.posterior['sigma_delta_top_boost'].values
        assert top_boost.min() > 0.0


class TestAZ2bStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ2b(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ2b(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_top_boost(self, data_dict):
        m = AZ2b(data_dict)
        m.build()
        for name in ('sigma_delta_top_boost', 'sigma_delta',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['sigma_delta_top_boost'].eval().shape == ()

    def test_sigma_delta_never_below_floor(self, data_dict):
        m = AZ2b(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['sigma_delta'], draws=20, random_seed=0)
        assert draws.min() >= AZ2b.sigma_delta_floor

    def test_boost_weight_is_smooth_not_binary(self, data_dict):
        # Regression test distinguishing AZ2b from AZ2: sigma_delta's
        # response to top_boost should include intermediate values near
        # the rank cutoff, not just the base formula or base+full boost.
        m = AZ2b(data_dict)
        m.build()
        D = data_dict['D']
        n_areas = data_dict['n_areas']
        with m.model:
            sigma_delta, top_boost = pm.draw(
                [m.model['sigma_delta'], m.model['sigma_delta_top_boost']],
                draws=1, random_seed=0)
        base = AZ2b.sigma_delta_floor + AZ2b.k_sigma_delta * np.abs(D)
        excess = sigma_delta - base
        weight = excess / top_boost
        # some areas should have weight strictly between 0 and 1 (not just
        # ~0 or ~1) -- proof the ramp is smooth, not a step.
        assert n_areas >= 8, 'test fixture too small to exercise the ramp'
        assert np.any((weight > 0.05) & (weight < 0.95))

    def test_var_names(self):
        assert set(AZ2b.var_names) == {
            'sigma_plan', 'sigma_ben', 'sigma_delta_top_boost'}


@pytest.mark.slow
class TestAZ2bSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ2b(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6


class TestAZ3Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ3(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ3(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_has_zero_sum_delta(self, data_dict):
        m = AZ3(data_dict)
        m.build()
        assert 'delta' in m.model.named_vars
        assert 'z' in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ3(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_noise_mixture_components(self, data_dict):
        m = AZ3(data_dict)
        m.build()
        for name in ('rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E',
                     'resp_noise_P', 'resp_noise_E',
                     'sigma_plan', 'sigma_ben', 'P_like', 'E_like'):
            assert name in m.model.named_vars
        assert m.model.named_vars['rho_P'].eval().shape == ()

    def test_sigma_noise_never_below_floor(self, data_dict):
        # Regression test for the exact failure mode that broke AZ0: a
        # noise branch with a scale free to shrink toward 0 has unbounded
        # density at the fixed point (0), and most P_obs/E_obs cells are
        # exactly 0. sigma_noise must never be able to reach the floor's
        # own value, let alone go below it.
        m = AZ3(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_noise_P'], m.model['sigma_noise_E']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ3.sigma_noise_floor

    def test_resp_noise_is_a_probability(self, data_dict):
        m = AZ3(data_dict)
        m.build()
        with m.model:
            resp = pm.draw(m.model['resp_noise_P'], draws=10, random_seed=0)
        assert resp.min() >= 0.0
        assert resp.max() <= 1.0

    def test_sigma_plan_ben_never_below_floor(self, data_dict):
        # Regression test for the ESS problem diagnosed on AZ3's real-data
        # run: an unfloored sigma_plan/sigma_ben collapsed to ~0.58/0.99,
        # an order of magnitude below every other AZ0a-family model's
        # ~7-9, creating small-scale funnel geometry. floor + HalfNormal
        # mirrors the already-proven sigma_noise_floor fix.
        m = AZ3(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_plan'], m.model['sigma_ben']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ3.sigma_obs_floor

    def test_var_names(self):
        assert set(AZ3.var_names) == {
            'sigma_plan', 'sigma_ben', 'rho_P', 'rho_E',
            'sigma_noise_P', 'sigma_noise_E'}


@pytest.mark.slow
class TestAZ3Sampling:
    def test_full_pipeline(self, data_dict):
        m = AZ3(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        sigma_noise_P = m.trace.posterior['sigma_noise_P'].values
        assert sigma_noise_P.min() >= AZ3.sigma_noise_floor

        resp_noise_P = m.trace.posterior['resp_noise_P'].values
        assert resp_noise_P.min() >= 0.0 and resp_noise_P.max() <= 1.0


class TestAZ4Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ4(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ4(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ4(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_all_three_pieces(self, data_dict):
        # One representative variable from each of the three combined
        # pieces must be present: AZ2b's smooth top-boost, AZ1b's
        # hierarchical lag (both sources), AZ3's floored noise mixture.
        m = AZ4(data_dict)
        m.build()
        for name in (
            'sigma_delta_top_boost', 'sigma_delta',                       # AZ2b piece
            'lag_P_mu_logit', 'lag_P_tau', 'lag_P_lambda_weights',        # AZ1b piece
            'lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights',
            'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E',           # AZ3 piece
            'resp_noise_P', 'resp_noise_E',
            'sigma_plan', 'sigma_ben', 'P_like', 'E_like',
        ):
            assert name in m.model.named_vars

    def test_sigma_plan_ben_never_below_floor(self, data_dict):
        m = AZ4(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_plan'], m.model['sigma_ben']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ4.sigma_obs_floor

    def test_sigma_noise_never_below_floor(self, data_dict):
        m = AZ4(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_noise_P'], m.model['sigma_noise_E']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ4.sigma_noise_floor

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ4(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_noise_mixture_applied_to_lag_convolved_mean_not_raw_z(self, data_dict):
        # AZ4's whole point of composition: the signal branch's mean must
        # be the LAG-CONVOLVED P_mean/E_mean, not raw z directly (unlike
        # AZ3, which had no lag structure to convolve). Verify indirectly:
        # P_like's pointwise log-lik must differ from what it would be if
        # z were used directly, for an area/year where lambda_weights
        # isn't degenerate at lag 0 with weight 1.
        m = AZ4(data_dict)
        m.build()
        with m.model:
            lambda_P = pm.draw(m.model['lag_P_lambda_weights'], draws=1, random_seed=0)
        # same-year (lag 0) weight should not be exactly 1.0 everywhere --
        # if it were, this test couldn't distinguish the two constructions.
        assert not np.allclose(lambda_P[:, 0], 1.0)

    def test_var_names(self):
        assert set(AZ4.var_names) == {
            'sigma_plan', 'sigma_ben', 'sigma_delta_top_boost',
            'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau',
            'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E'}


@pytest.mark.slow
class TestAZ4Sampling:
    def test_full_pipeline(self, data_dict):
        m = AZ4(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        sigma_plan = m.trace.posterior['sigma_plan'].values
        assert sigma_plan.min() >= AZ4.sigma_obs_floor

        sigma_noise_P = m.trace.posterior['sigma_noise_P'].values
        assert sigma_noise_P.min() >= AZ4.sigma_noise_floor


class TestAZ4bStructure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ4b(data_dict).build(), pm.Model)

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ4b(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_tau_never_exceeds_cap(self, data_dict):
        m = AZ4b(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['lag_P_tau'], m.model['lag_E_tau']],
                draws=200, random_seed=0)
        for d in draws:
            assert d.min() >= 0.0
            assert d.max() <= AZ4b.tau_cap

    def test_has_all_three_pieces(self, data_dict):
        m = AZ4b(data_dict)
        m.build()
        for name in (
            'sigma_delta_top_boost', 'sigma_delta',
            'lag_P_mu_logit', 'lag_P_tau_frac', 'lag_P_tau', 'lag_P_lambda_weights',
            'lag_E_mu_logit', 'lag_E_tau_frac', 'lag_E_tau', 'lag_E_lambda_weights',
            'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E',
            'resp_noise_P', 'resp_noise_E',
            'sigma_plan', 'sigma_ben', 'P_like', 'E_like',
        ):
            assert name in m.model.named_vars

    def test_var_names(self):
        assert set(AZ4b.var_names) == set(AZ4.var_names)


@pytest.mark.slow
class TestAZ4bSampling:
    def test_full_pipeline(self, data_dict):
        m = AZ4b(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        tau_P = m.trace.posterior['lag_P_tau'].values
        assert tau_P.max() <= AZ4b.tau_cap


class TestAZ5Structure:
    def test_build_returns_model(self, data_dict):
        assert isinstance(AZ5(data_dict).build(), pm.Model)

    def test_no_census_obs(self, data_dict):
        m = AZ5(data_dict)
        m.build()
        assert 'census_obs' not in m.model.named_vars

    def test_z_sums_to_d_exactly(self, data_dict):
        m = AZ5(data_dict)
        m.build()
        with m.model:
            z_draws = pm.draw(m.model['z'], draws=5, random_seed=0)
        resid = np.abs(z_draws.sum(axis=-1) - data_dict['D'][None, :])
        assert resid.max() < 1e-6

    def test_has_both_pieces(self, data_dict):
        # One representative variable from each of the two combined
        # pieces: AZ1g's P-only regularized-horseshoe lag, AZ3's floored
        # noise mixture (applied to P_mean for P, raw z for E).
        m = AZ5(data_dict)
        m.build()
        for name in (
            'lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_local_lambda',   # AZ1g piece
            'lag_P_tau', 'lag_P_raw_offset', 'lag_P_lambda_weights',
            'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E',           # AZ3 piece
            'resp_noise_P', 'resp_noise_E',
            'sigma_plan', 'sigma_ben', 'P_like', 'E_like',
        ):
            assert name in m.model.named_vars

    def test_no_e_lag_machinery(self, data_dict):
        # Like AZ1d/AZ1g: only P gets a lag mechanism, E stays same-year.
        m = AZ5(data_dict)
        m.build()
        for name in ('lag_E_mu_logit', 'lag_E_tau', 'lag_E_lambda_weights'):
            assert name not in m.model.named_vars

    def test_sigma_plan_ben_never_below_floor(self, data_dict):
        m = AZ5(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_plan'], m.model['sigma_ben']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ5.sigma_obs_floor

    def test_sigma_noise_never_below_floor(self, data_dict):
        m = AZ5(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(
                [m.model['sigma_noise_P'], m.model['sigma_noise_E']],
                draws=50, random_seed=0)
        for d in draws:
            assert d.min() >= AZ5.sigma_noise_floor

    def test_lambda_weights_are_simplices(self, data_dict):
        m = AZ5(data_dict)
        m.build()
        with m.model:
            draws = pm.draw(m.model['lag_P_lambda_weights'], draws=5, random_seed=0)
        assert draws.min() >= 0.0
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    def test_noise_mixture_applied_to_lag_convolved_mean_for_p(self, data_dict):
        # Mirrors AZ4's own check: P's signal branch must use the
        # LAG-CONVOLVED P_mean, not raw z directly.
        m = AZ5(data_dict)
        m.build()
        with m.model:
            lambda_P = pm.draw(m.model['lag_P_lambda_weights'], draws=1, random_seed=0)
        assert not np.allclose(lambda_P[:, 0], 1.0)

    def test_var_names(self):
        assert set(AZ5.var_names) == {
            'sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau',
            'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E'}


@pytest.mark.slow
class TestAZ5Sampling:
    def test_full_pipeline(self, data_dict):
        m = AZ5(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8, random_seed=0)

        assert 'z' in m.trace.posterior
        assert 'P_like' in m.trace.log_likelihood
        assert 'E_like' in m.trace.log_likelihood

        z = m.trace.posterior['z'].values
        D = data_dict['D']
        resid = np.abs(z.sum(axis=-1) - D[None, None, :])
        assert resid.max() < 1e-6

        sigma_plan = m.trace.posterior['sigma_plan'].values
        assert sigma_plan.min() >= AZ5.sigma_obs_floor

        sigma_noise_P = m.trace.posterior['sigma_noise_P'].values
        assert sigma_noise_P.min() >= AZ5.sigma_noise_floor


# ── Sampling integration tests (slow — run with pytest -m slow) ──────────────

@pytest.mark.slow
class TestSamplingPipeline:
    """
    Run the full sample() → compute_log_likelihood pipeline on M0 with a
    tiny draw count. Catches bugs where the trace is missing groups (e.g.
    log_likelihood) that downstream functions like az.compare require.
    """

    def test_trace_has_log_likelihood(self, data_dict):
        m = M0(data_dict)
        m.sample(use_nutpie=False, draws=50, tune=50, chains=1, cores=1,
                 target_accept=0.8)
        assert hasattr(m.trace, 'log_likelihood'), \
            "trace is missing log_likelihood group — az.compare will fail"

    def test_pointwise_log_likelihood_attached_for_potential_based_models(self, data_dict):
        # M13's likelihood is built from pm.Potential (a marginalised
        # mixture), not an observed RV -- pm.compute_log_likelihood()
        # alone can't derive per-observation log-likelihood for it.
        # Regression test for "no log likelihood data named P_like
        # found": the builder-exposed '*_pointwise' Deterministics must
        # get attached to the log_likelihood group under their
        # un-suffixed name.
        m = M13(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8)
        assert 'P_like' in m.trace.log_likelihood.data_vars
        assert 'E_like' in m.trace.log_likelihood.data_vars

    def test_pointwise_log_likelihood_survives_save_reload(self, data_dict, tmp_path):
        # Regression test: xarray DataTree attribute-style assignment
        # (trace.log_likelihood = ...) looks correct on an in-memory read
        # right after assignment, but doesn't update the node to_netcdf()
        # serialises from -- the group silently comes back empty after a
        # save()/load() round trip. _attach_pointwise_log_likelihood must
        # use item-style assignment (trace['log_likelihood'] = ...) instead.
        m = M13(data_dict)
        m.sample(use_nutpie=False, draws=20, tune=20, chains=1, cores=1,
                 target_accept=0.8)
        m.save(results_dir=str(tmp_path))

        m.load(results_dir=str(tmp_path))
        assert 'P_like' in m.trace.log_likelihood.data_vars
        assert 'E_like' in m.trace.log_likelihood.data_vars

    def test_model_comparison_runs_after_sampling(self, data_dict):
        from housing_projections.analysis import compute_model_comparison

        traces = {}
        for ModelClass in (M0, M1):
            m = ModelClass(data_dict)
            m.sample(use_nutpie=False, draws=100, tune=50, chains=2, cores=1,
                     target_accept=0.8)
            traces[ModelClass.name] = m.trace

        result = compute_model_comparison(traces, verbose=False)
        assert set(result.index) == {'M0', 'M1'}
