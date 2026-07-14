"""
Model registry — the single source of truth for "every implemented model".

Model classes themselves live in `m_family.py` (M0-M16) and `az_family.py` (AZ0-AZ4b); shared
construction pieces live in `builders.py`. This module just imports all three, re-exports
every name from them (so existing `from housing_projections.models.models import M0` /
`_build_...` imports keep working unchanged), and assembles `ALL_MODELS` below —
`housing_projections.models` and the CLI both derive their registries from `ALL_MODELS`
rather than re-listing models.
"""
from .az_family import (
    AZ0,
    AZ2,
    AZ3,
    AZ4,
    AZ5,
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
)
from .base import DwellingModel
from .builders import (
    _build_agreement_gated_likelihood,
    _build_asymmetric_missingness,
    _build_backward_reallocation_likelihood,
    _build_backward_reallocation_likelihood_2way,
    _build_capture_rate,
    _build_census_constraint,
    _build_fixed_lag,
    _build_hierarchical_lag,
    _build_hierarchical_lag_capped,
    _build_hierarchical_lag_horseshoe,
    _build_hierarchical_lag_marginalized,
    _build_hierarchical_lag_pinned,
    _build_hierarchical_lag_regularized_horseshoe,
    _build_hierarchical_lag_regularized_horseshoe_v2,
    _build_independent_agreement_gated_likelihood,
    _build_lag,
    _build_noise_mixture_likelihood,
    _build_planning_likelihood_marginalized_lag,
    _build_planning_likelihood_simple,
    _build_planning_likelihood_zeroinflated,
    _build_pre_inference,
    _build_spatial_misallocation,
    _build_temporal_reallocation_likelihood,
    _build_temporal_reallocation_likelihood_marginalizable,
    _build_z_prior,
    _build_z_prior_hierarchical,
    _build_z_prior_hierarchical_borough,
    _build_z_prior_profile_library,
    _build_z_prior_profile_library_horseshoe,
    _build_zero_sum_profile_library,
    _build_zero_sum_z_prior,
    _build_zero_sum_z_prior_banded,
    _build_zero_sum_z_prior_top_boost,
    _build_zero_sum_z_prior_top_boost_smooth,
)
from .m_family import M0, M1, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M0h, M1h

# ── Model registry (single source of truth) ────────────────────────────────
#
# Every implemented model class is listed here exactly once. Adding a new
# model means adding it to this list — housing_projections.models re-exports
# it automatically, and the CLI / notebooks discover it through ALL_MODELS
# rather than needing their own copy of this list.
#
# NOTE: this is a plain dict literal, not a `{cls.name: cls for cls in [...]}`
# comprehension — mypy resolves `name` (an abstract property on DwellingModel,
# overridden as a plain str attribute on each subclass) to `str` per literal
# class reference, but widens it to the unbound property getter once the
# classes are joined into a single iterable, which breaks every `ALL_MODELS
# [name]` lookup downstream.
ALL_MODELS: dict[str, type[DwellingModel]] = {
    M0.name: M0, M0h.name: M0h, M1.name: M1, M1h.name: M1h,
    M5.name: M5, M6.name: M6, M7.name: M7, M8.name: M8,
    M9.name: M9, M10.name: M10, M11.name: M11, M12.name: M12,
    M13.name: M13, M14.name: M14, M15.name: M15, M16.name: M16,
    AZ0.name: AZ0, AZ0a.name: AZ0a, AZ0b.name: AZ0b,
    AZ1a.name: AZ1a, AZ1b.name: AZ1b, AZ1c.name: AZ1c,
    AZ1d.name: AZ1d, AZ1e.name: AZ1e, AZ1f.name: AZ1f, AZ1g.name: AZ1g, AZ1h.name: AZ1h,
    AZ2.name: AZ2, AZ2b.name: AZ2b, AZ3.name: AZ3,
    AZ4.name: AZ4, AZ4b.name: AZ4b, AZ5.name: AZ5,
}

__all__ = [
    "ALL_MODELS", *ALL_MODELS,
    "_build_agreement_gated_likelihood",
    "_build_asymmetric_missingness",
    "_build_backward_reallocation_likelihood",
    "_build_backward_reallocation_likelihood_2way",
    "_build_capture_rate",
    "_build_census_constraint",
    "_build_fixed_lag",
    "_build_hierarchical_lag",
    "_build_hierarchical_lag_capped",
    "_build_hierarchical_lag_horseshoe",
    "_build_hierarchical_lag_marginalized",
    "_build_hierarchical_lag_pinned",
    "_build_hierarchical_lag_regularized_horseshoe",
    "_build_hierarchical_lag_regularized_horseshoe_v2",
    "_build_independent_agreement_gated_likelihood",
    "_build_lag",
    "_build_noise_mixture_likelihood",
    "_build_planning_likelihood_marginalized_lag",
    "_build_planning_likelihood_simple",
    "_build_planning_likelihood_zeroinflated",
    "_build_pre_inference",
    "_build_spatial_misallocation",
    "_build_temporal_reallocation_likelihood",
    "_build_temporal_reallocation_likelihood_marginalizable",
    "_build_z_prior",
    "_build_z_prior_hierarchical",
    "_build_z_prior_hierarchical_borough",
    "_build_z_prior_profile_library",
    "_build_z_prior_profile_library_horseshoe",
    "_build_zero_sum_profile_library",
    "_build_zero_sum_z_prior",
    "_build_zero_sum_z_prior_banded",
    "_build_zero_sum_z_prior_top_boost",
    "_build_zero_sum_z_prior_top_boost_smooth",
]
