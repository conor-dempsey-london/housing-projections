"""
Tests for housing_projections.multimodality.

The classifier fixture (`classifier_trace`) hand-builds five areas, one per category
(hard_genuine, stuck_fixable, round_tripping, mixed, needs_review) with deterministic
lambda_weights/log_likelihood values chosen so each area's r-hat/switch-rate/gap land
unambiguously in the intended bucket — verified numerically against this exact
construction before being locked in as test assertions (not asserted from first
principles), the same discipline the classifier itself was built to encourage.
"""
import arviz as az
import numpy as np
import pytest

from housing_projections.diagnostics import diagnostics_summary
from housing_projections.multimodality import (
    CAT_HARD_GENUINE,
    CAT_NOT_MULTIMODAL,
    CAT_STUCK_FIXABLE,
    _cluster_chain_means,
    _discover_modes_from_chain_means,
    _nearest_mode_labels,
    _switch_rate_from_dominant,
    adjusted_diagnostics_report,
    adjusted_diagnostics_summary,
    classify_multimodality,
    classify_scalar_multimodality,
    compute_loglik_gap,
    compute_logp_gap,
    compute_switch_rates,
    multimodality_report,
    resolve_stuck_areas,
    verify_resolution,
)

N_CHAINS = 8
N_DRAWS  = 1500
N_CATS   = 3
N_AREAS  = 5
AREA_CODES = [f"E{i:08d}" for i in range(N_AREAS)]
# positional indices used only for array construction below
IDX_HARD_GENUINE, IDX_STUCK_FIXABLE, IDX_ROUND_TRIPPING, IDX_MIXED, IDX_NEEDS_REVIEW = range(5)
# area CODES (what classify_multimodality actually reports, once 'area' coords exist —
# matching real traces, where the area coord always carries LSOA codes, not bare indices)
(AREA_HARD_GENUINE, AREA_STUCK_FIXABLE, AREA_ROUND_TRIPPING,
 AREA_MIXED, AREA_NEEDS_REVIEW) = AREA_CODES


def _onehot(cat, n_cats=N_CATS, hi=0.9):
    v = np.full(n_cats, (1 - hi) / (n_cats - 1))
    v[cat] = hi
    return v


def _build_classifier_arrays():
    """(lw, ll) for the five hand-built areas (one per category) shared by every
    fixture below — see the module docstring for how each area's values were chosen."""
    rng = np.random.default_rng(11)
    lw = np.zeros((N_CHAINS, N_DRAWS, N_AREAS, N_CATS))
    ll = np.zeros((N_CHAINS, N_DRAWS, N_AREAS, 1))

    # hard_genuine: chains 0-3 locked cat0, chains 4-7 locked cat1, tied log-likelihood
    for c in range(N_CHAINS):
        cat = 0 if c < 4 else 1
        lw[c, :, IDX_HARD_GENUINE, :] = _onehot(cat)
        ll[c, :, IDX_HARD_GENUINE, 0] = -50.0 if c < 4 else -50.3

    # stuck_fixable: chains 0-6 locked cat1 (best), chain 7 locked cat0 (decisively worse)
    for c in range(N_CHAINS):
        cat = 1 if c < 7 else 0
        lw[c, :, IDX_STUCK_FIXABLE, :] = _onehot(cat)
        ll[c, :, IDX_STUCK_FIXABLE, 0] = -40.0 if c < 7 else -55.0

    # round_tripping: Markov-chain switching (mean dwell ~30 draws), mild per-chain bias —
    # matches AZ1d's real E01035649 profile (r-hat ~1.02-1.04, switch rate ~2-4%), NOT a
    # perfectly periodic pattern (which produces degenerate, near-singular r-hat instead
    # of the mild elevation real round-tripping areas actually show).
    for c in range(N_CHAINS):
        p_1to0 = 0.033 + 0.004 * (c - 3.5) / 3.5
        p_0to1 = 0.033 - 0.004 * (c - 3.5) / 3.5
        cats = np.zeros(N_DRAWS, dtype=int)
        cats[0] = c % 2
        for d in range(1, N_DRAWS):
            if cats[d - 1] == 0:
                cats[d] = 1 if rng.random() < p_0to1 else 0
            else:
                cats[d] = 0 if rng.random() < p_1to0 else 1
        for d in range(N_DRAWS):
            lw[c, d, IDX_ROUND_TRIPPING, :] = _onehot(cats[d])
        ll[c, :, IDX_ROUND_TRIPPING, 0] = -45.0 + rng.normal(0, 0.5, size=N_DRAWS)

    # mixed: chains 0-4 locked cat1 (best), chains 5-6 locked cat2 (tied with best),
    # chain 7 locked cat0 (decisively worse straggler)
    for c in range(N_CHAINS):
        cat = 1 if c < 5 else (2 if c < 7 else 0)
        lw[c, :, IDX_MIXED, :] = _onehot(cat)
        ll[c, :, IDX_MIXED, 0] = -45.0 if c < 5 else (-45.5 if c < 7 else -70.0)

    # needs_review: chains 0-3 locked cat0, chains 4-7 locked cat1, gap=5.0 nats — deliberately
    # between gap_tied_threshold (2.0) and gap_decisive_threshold (10.0)
    for c in range(N_CHAINS):
        cat = 0 if c < 4 else 1
        lw[c, :, IDX_NEEDS_REVIEW, :] = _onehot(cat)
        ll[c, :, IDX_NEEDS_REVIEW, 0] = -30.0 if c < 4 else -35.0

    return lw, ll


@pytest.fixture(scope="module")
def classifier_trace():
    lw, ll = _build_classifier_arrays()

    # Matches the real AZ1d trace's actual dim wiring: lag_P_lambda_weights/P_like get
    # their own ANONYMOUS PyMC dims (not tied to the 'area' coord) since
    # _build_hierarchical_lag's pm.Deterministic doesn't pass dims= explicitly — the
    # 'area' coord instead lives on 'z' (dims=('area', 'year') via _default_coords), and
    # classify_multimodality looks it up there separately. A dummy 'z' variable
    # reproduces that real structure here rather than giving lag_P_lambda_weights its
    # own area-linked dim, which classify_multimodality is NOT written to expect (and
    # never sees in production).
    z_dummy = np.zeros((N_CHAINS, N_DRAWS, N_AREAS, 1))
    return az.from_dict(
        {"posterior": {"lag_P_lambda_weights": lw, "z": z_dummy},
         "log_likelihood": {"P_like": ll}},
        coords={"area": AREA_CODES},
        dims={"z": ["area", "year"]},
    )


@pytest.fixture(scope="module")
def classifier_trace_with_sample_stats():
    """Same five areas as `classifier_trace`, plus a `sample_stats.diverging` group —
    diagnostics.diagnostics_summary (the base adjusted_diagnostics_summary wraps) needs
    it, unlike the plain classifier/report tests which never go through that path."""
    lw, ll = _build_classifier_arrays()
    z_dummy = np.zeros((N_CHAINS, N_DRAWS, N_AREAS, 1))
    diverging = np.zeros((N_CHAINS, N_DRAWS), dtype=bool)
    return az.from_dict(
        {"posterior": {"lag_P_lambda_weights": lw, "z": z_dummy},
         "log_likelihood": {"P_like": ll},
         "sample_stats": {"diverging": diverging}},
        coords={"area": AREA_CODES},
        dims={"z": ["area", "year"]},
    )


@pytest.fixture(scope="module")
def classification_df(classifier_trace):
    return classify_multimodality(classifier_trace, "lag_P_lambda_weights", "P_like")


class TestComputeSwitchRates:
    def test_locked_chains_never_switch(self, classifier_trace):
        rate, _ = compute_switch_rates(classifier_trace, "lag_P_lambda_weights")
        assert rate[:, IDX_HARD_GENUINE].max() == 0.0
        assert rate[:, IDX_STUCK_FIXABLE].max() == 0.0

    def test_round_tripping_area_has_high_switch_rate(self, classifier_trace):
        rate, _ = compute_switch_rates(classifier_trace, "lag_P_lambda_weights")
        assert rate[:, IDX_ROUND_TRIPPING].mean() > 0.02


class TestComputeLoglikGap:
    def test_hard_genuine_gap_is_small(self, classifier_trace):
        _, dominant = compute_switch_rates(classifier_trace, "lag_P_lambda_weights")
        result = compute_loglik_gap(classifier_trace, dominant, IDX_HARD_GENUINE, "P_like")
        assert result["gaps"][1] == pytest.approx(0.3, abs=1e-6)

    def test_stuck_fixable_gap_is_large(self, classifier_trace):
        _, dominant = compute_switch_rates(classifier_trace, "lag_P_lambda_weights")
        result = compute_loglik_gap(classifier_trace, dominant, IDX_STUCK_FIXABLE, "P_like")
        assert result["best_category"] == 1
        assert result["gaps"][0] == pytest.approx(15.0, abs=1e-6)

    def test_mixed_area_has_one_tied_and_one_decisive_gap(self, classifier_trace):
        _, dominant = compute_switch_rates(classifier_trace, "lag_P_lambda_weights")
        result = compute_loglik_gap(classifier_trace, dominant, IDX_MIXED, "P_like")
        assert result["best_category"] == 1
        assert result["gaps"][2] == pytest.approx(0.5, abs=1e-6)
        assert result["gaps"][0] == pytest.approx(25.0, abs=1e-6)


class TestClusterChainMeans:
    """
    Standardized-gap clustering for the generic (non-*_lambda_weights) mode-discovery path —
    supersedes an earlier KDE-peak-on-chain-means approach that was tried and rejected during
    implementation: scipy's default KDE bandwidth over-smoothed an obvious cluster split into
    one lump, while a hand-tuned smaller bandwidth that DID separate it also false-positived on
    pure unimodal noise and still missed a minority 1-vs-N chain split. These test cases are the
    exact synthetic scenarios that comparison was run against.
    """

    def _n_clusters(self, chain_means, within_std, sigma_threshold=3.0):
        return len(_cluster_chain_means(chain_means, within_std, sigma_threshold))

    def test_obvious_half_split_found(self):
        means = np.array([0.0, 0.05, -0.05, 0.02, 5.0, 5.05, 4.95, 5.02])
        assert self._n_clusters(means, within_std=1.0) == 2

    def test_minority_split_found(self):
        # a 3-vs-1 split -- the exact case KDE-on-chain-means missed even with a tuned
        # bandwidth (the minority cluster's KDE bump was too small relative to prominence).
        means = np.array([0.0, 0.1, -0.1, 5.0])
        assert self._n_clusters(means, within_std=1.0) == 2

    def test_unimodal_noise_not_split(self):
        # KDE-on-chain-means with a small bandwidth falsely reported 2 clusters here.
        means = np.array([0.03, 0.08, 0.03, -0.13, 0.09, 0.04, -0.05, 0.06])
        assert self._n_clusters(means, within_std=1.0) == 1

    def test_linear_spread_not_split(self):
        means = np.linspace(-2, 2, 8)
        assert self._n_clusters(means, within_std=1.0) == 1

    def test_gap_smaller_than_noise_not_split(self):
        # gap (2) below the sigma_threshold (3x within_std=1) -- not cleanly separated.
        means = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
        assert self._n_clusters(means, within_std=1.0) == 1

    def test_zero_within_std_never_divides_by_zero(self):
        # every chain literally identical -- within_std=0 must not raise, and a single
        # degenerate cluster (or as many as there are distinct values) is fine either way,
        # just must not crash.
        means = np.array([1.0, 1.0, 1.0, 1.0])
        assert self._n_clusters(means, within_std=0.0) == 1


class TestDiscoverModesFromChainMeans:
    def _values(self, means, draw_std=1.0, n_draws=200, seed=0):
        rng = np.random.default_rng(seed)
        return np.stack([rng.normal(m, draw_std, n_draws) for m in means])

    def test_returns_two_locations_for_a_real_split(self):
        values = self._values([0, 0, 0, 0, 5, 5, 5, 5])
        locations = _discover_modes_from_chain_means(values)
        assert len(locations) == 2
        assert locations.min() < 1.0
        assert locations.max() > 4.0

    def test_returns_empty_for_unimodal_data(self):
        values = self._values([0, 0, 0, 0, 0, 0, 0, 0])
        locations = _discover_modes_from_chain_means(values)
        assert len(locations) == 0


class TestNearestModeLabels:
    def test_labels_match_nearest_location(self):
        values = np.array([[0.1, 0.2, 4.9], [5.1, -0.1, 5.0]])
        locations = np.array([0.0, 5.0])
        labels = _nearest_mode_labels(values, locations)
        assert labels.tolist() == [[0, 0, 1], [1, 0, 1]]


class TestSwitchRateFromDominant:
    def test_no_switches_gives_zero_rate(self):
        dominant = np.zeros((3, 10), dtype=int)
        rate = _switch_rate_from_dominant(dominant)
        assert (rate == 0.0).all()

    def test_every_draw_switching_gives_full_rate(self):
        dominant = np.tile([0, 1], (2, 5))
        rate = _switch_rate_from_dominant(dominant)
        assert (rate == 1.0).all()

    def test_single_draw_gives_zero_not_a_crash(self):
        dominant = np.zeros((2, 1), dtype=int)
        rate = _switch_rate_from_dominant(dominant)
        assert (rate == 0.0).all()


class TestComputeLogpGap:
    def _trace_with_logp(self, logp_by_chain):
        n_chains, n_draws = logp_by_chain.shape
        return az.from_dict(
            {"posterior": {"dummy": np.zeros((n_chains, n_draws))},
             "sample_stats": {"logp": logp_by_chain}})

    def test_gap_expressed_in_sigma_units_not_nats(self):
        rng = np.random.default_rng(1)
        n_draws = 300
        # 4 chains at logp~-40 (std 0.5), 4 at logp~-50 (std 0.5) -- a 10-nat gap, but
        # within_std=0.5 -> expect a gap of ~20 SIGMA, not 10 nats.
        logp = np.stack([rng.normal(-40.0, 0.5, n_draws) for _ in range(4)]
                        + [rng.normal(-50.0, 0.5, n_draws) for _ in range(4)])
        trace = self._trace_with_logp(logp)
        dominant = np.array([[0] * n_draws] * 4 + [[1] * n_draws] * 4)
        result = compute_logp_gap(trace, dominant)
        assert result["best_category"] == 0
        assert result["gaps"][1] == pytest.approx(20.0, rel=0.2)

    def test_no_logp_or_lp_returns_none(self):
        trace = az.from_dict({"posterior": {"dummy": np.zeros((2, 10))}})
        dominant = np.zeros((2, 10), dtype=int)
        result = compute_logp_gap(trace, dominant)
        assert result["best_category"] is None

    def test_falls_back_to_lp_when_logp_absent(self):
        rng = np.random.default_rng(2)
        n_chains, n_draws = 4, 200
        lp = np.stack([rng.normal(-10.0, 0.2, n_draws) for _ in range(2)]
                      + [rng.normal(-20.0, 0.2, n_draws) for _ in range(2)])
        trace = az.from_dict(
            {"posterior": {"dummy": np.zeros((n_chains, n_draws))},
             "sample_stats": {"lp": lp}})
        dominant = np.array([[0] * n_draws] * 2 + [[1] * n_draws] * 2)
        result = compute_logp_gap(trace, dominant)
        assert result["best_category"] == 0


class TestClassifyScalarMultimodality:
    def _trace(self, values, logp=None, extra_posterior=None):
        # values may be (chain, draw) for a true scalar or (chain, draw, *extra) for a
        # vector-valued variable -- only sample_stats needs the plain (chain, draw) shape.
        posterior = {"my_scalar": values}
        if extra_posterior:
            posterior.update(extra_posterior)
        groups = {"posterior": posterior}
        if logp is not None:
            groups["sample_stats"] = {"logp": logp}
        return az.from_dict(groups)

    def test_decisive_gap_classified_stuck_fixable(self):
        rng = np.random.default_rng(4)
        n_draws = 500
        values = np.stack([rng.normal(mu, 1.0, n_draws) for mu in [0] * 6 + [8] * 2])
        logp = np.stack([rng.normal(-40.0, 0.5, n_draws) for _ in range(6)]
                        + [rng.normal(-70.0, 0.5, n_draws) for _ in range(2)])
        trace = self._trace(values, logp=logp)

        df = classify_scalar_multimodality(trace, ["my_scalar"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["var"] == "my_scalar"
        assert row["cell"] == ()
        assert row["category"] == CAT_STUCK_FIXABLE

    def test_tied_gap_classified_hard_genuine(self):
        rng = np.random.default_rng(3)
        n_draws = 500
        values = np.stack([rng.normal(mu, 1.0, n_draws) for mu in [0] * 4 + [5] * 4])
        logp = np.stack([rng.normal(-50.0, 0.5, n_draws) for _ in range(4)]
                        + [rng.normal(-50.2, 0.5, n_draws) for _ in range(4)])
        trace = self._trace(values, logp=logp)

        df = classify_scalar_multimodality(trace, ["my_scalar"])
        assert len(df) == 1
        assert df.iloc[0]["category"] == CAT_HARD_GENUINE

    def test_no_chain_clustering_classified_not_multimodal(self):
        rng = np.random.default_rng(6)
        n_draws = 500
        # bad r-hat via a linear per-chain offset spread, not a genuine 2-cluster split.
        values = np.stack([rng.normal(mu, 1.0, n_draws) for mu in np.linspace(-3, 3, 8)])
        logp = rng.normal(-50.0, 0.5, size=(8, n_draws))
        trace = self._trace(values, logp=logp)

        df = classify_scalar_multimodality(trace, ["my_scalar"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["category"] == CAT_NOT_MULTIMODAL
        assert row["best_category"] is None

    def test_healthy_variable_not_flagged_at_all(self):
        rng = np.random.default_rng(8)
        values = rng.normal(0, 1.0, size=(8, 500))
        trace = self._trace(values)
        df = classify_scalar_multimodality(trace, ["my_scalar"])
        assert df.empty

    def test_vector_variable_checks_each_cell_independently(self):
        # shape (chain, draw, 2) -- one cell flagged (decisive split), one healthy.
        rng = np.random.default_rng(9)
        n_draws = 500
        flagged_cell = np.stack([rng.normal(mu, 1.0, n_draws) for mu in [0] * 6 + [8] * 2])
        healthy_cell = rng.normal(0, 1.0, size=(8, n_draws))
        values = np.stack([flagged_cell, healthy_cell], axis=-1)  # (chain, draw, 2)
        logp = np.stack([rng.normal(-40.0, 0.5, n_draws) for _ in range(6)]
                        + [rng.normal(-70.0, 0.5, n_draws) for _ in range(2)])
        trace = self._trace(values, logp=logp)

        df = classify_scalar_multimodality(trace, ["my_scalar"])
        assert len(df) == 1
        assert df.iloc[0]["cell"] == (0,)

    def test_missing_var_name_silently_skipped(self):
        trace = self._trace(np.zeros((4, 50)))
        df = classify_scalar_multimodality(trace, ["does_not_exist"])
        assert df.empty


class TestClassifyMultimodality:
    def test_all_five_areas_flagged(self, classification_df):
        assert len(classification_df) == 5

    def test_hard_genuine_classified_correctly(self, classification_df):
        row = classification_df[classification_df["area"] == AREA_HARD_GENUINE].iloc[0]
        assert row["category"] == "hard_genuine"

    def test_stuck_fixable_classified_correctly(self, classification_df):
        row = classification_df[classification_df["area"] == AREA_STUCK_FIXABLE].iloc[0]
        assert row["category"] == "stuck_fixable"
        assert row["best_category"] == 1

    def test_round_tripping_classified_correctly(self, classification_df):
        row = classification_df[classification_df["area"] == AREA_ROUND_TRIPPING].iloc[0]
        assert row["category"] == "round_tripping"
        assert row["mean_switch_rate"] > 0.02

    def test_mixed_classified_correctly(self, classification_df):
        row = classification_df[classification_df["area"] == AREA_MIXED].iloc[0]
        assert row["category"] == "mixed"

    def test_needs_review_classified_correctly(self, classification_df):
        row = classification_df[classification_df["area"] == AREA_NEEDS_REVIEW].iloc[0]
        assert row["category"] == "needs_review"

    def test_no_flags_returns_empty_dataframe_with_expected_columns(self):
        # Needs enough draws that i.i.d. noise doesn't spuriously cross the r-hat
        # threshold by chance (verified: 50 draws/4 chains did, occasionally; 2000
        # draws is comfortably stable) — hierarchical_mode_summary should correctly
        # find nothing to flag here, not an artifact of too little data.
        n_chains, n_draws, n_areas, n_cats = 4, 2000, 2, 2
        lw = np.full((n_chains, n_draws, n_areas, n_cats), 0.5)
        rng = np.random.default_rng(0)
        lw += rng.normal(0, 0.01, size=lw.shape)
        trace = az.from_dict({"posterior": {"lag_P_lambda_weights": lw}})
        df = classify_multimodality(trace, "lag_P_lambda_weights", "P_like")
        assert df.empty
        assert list(df.columns) == [
            "area", "max_rhat", "mean_switch_rate", "n_pure_chains", "n_mixed_chains",
            "best_category", "gaps", "category", "recommended_action",
        ]


class TestResolveStuckAreas:
    """Unit tests for resolve_stuck_areas' pure seeding logic, via a duck-typed fake
    model that records its sample() call instead of actually running PyMC — the
    end-to-end sampling path is exercised separately in TestResolveStuckAreasIntegration.
    """

    class _FakeModel:
        def __init__(self, data):
            self.data = data
            self.n_lags = 3
            self.sample_kwargs_received = None
            self.trace = "FAKE_TRACE"

        def sample(self, **kwargs):
            self.sample_kwargs_received = kwargs

    def _fake_data(self, n_areas=5):
        return {"gdf": _FakeGdf([f"E{i:08d}" for i in range(n_areas)]), "n_areas": n_areas}

    def test_seeds_only_stuck_fixable_areas(self, classification_df):
        captured = {}

        def factory(data):
            m = self._FakeModel(data)
            captured["model"] = m
            return m

        trace, seeded = resolve_stuck_areas(
            factory, self._fake_data(), classification_df, "lag_P_lambda_weights", chains=16)

        assert seeded == [AREA_STUCK_FIXABLE]
        init_mean = captured["model"].sample_kwargs_received["init_mean"]
        raw_offset = init_mean["lag_P_raw_offset"]
        # best_category=1 -> free_dim=0 seeded positive, other free_dim seeded negative
        assert raw_offset[IDX_STUCK_FIXABLE, 0] == 3.0
        assert raw_offset[IDX_STUCK_FIXABLE, 1] == -1.0
        # untouched areas (hard_genuine, round_tripping, mixed, needs_review) stay at zero
        for idx in (IDX_HARD_GENUINE, IDX_ROUND_TRIPPING, IDX_MIXED, IDX_NEEDS_REVIEW):
            assert (raw_offset[idx] == 0.0).all()

    def test_returns_none_when_no_stuck_fixable_areas(self):
        import pandas as pd
        empty_df = pd.DataFrame(columns=["area", "category", "best_category"])
        trace, seeded = resolve_stuck_areas(
            self._FakeModel, self._fake_data(), empty_df, "lag_P_lambda_weights")
        assert trace is None
        assert seeded == []


class TestVerifyResolution:
    def _make_resolved_trace(self, agreement_frac):
        """16 chains, one area, `agreement_frac` of them locked into category 1, the
        rest locked into category 0 — mimics a post-reseed check on a real trace."""
        n_chains, n_draws, n_cats = 16, 50, 3
        n_agree = round(n_chains * agreement_frac)
        lw = np.zeros((n_chains, n_draws, 1, n_cats))
        for c in range(n_chains):
            lw[c, :, 0, :] = _onehot(1 if c < n_agree else 0)
        return az.from_dict(
            {"posterior": {"lag_P_lambda_weights": lw}},
            coords={"area": [AREA_STUCK_FIXABLE]},
            dims={"lag_P_lambda_weights": ["area", "category"]},
        )

    def test_high_agreement_counts_as_resolved(self):
        trace = self._make_resolved_trace(agreement_frac=1.0)
        result = verify_resolution(trace, "lag_P_lambda_weights", [AREA_STUCK_FIXABLE])
        row = result.iloc[0]
        assert row["chain_agreement_frac"] == 1.0
        assert row["resolved"]

    def test_low_agreement_not_resolved(self):
        trace = self._make_resolved_trace(agreement_frac=0.5)
        result = verify_resolution(trace, "lag_P_lambda_weights", [AREA_STUCK_FIXABLE])
        row = result.iloc[0]
        assert row["chain_agreement_frac"] == pytest.approx(0.5)
        assert not row["resolved"]

    def test_ignores_areas_not_in_trace(self):
        trace = self._make_resolved_trace(agreement_frac=1.0)
        result = verify_resolution(trace, "lag_P_lambda_weights",
                                   [AREA_STUCK_FIXABLE, "E99999999"])
        assert len(result) == 1


class _FakeGdf:
    """Minimal stand-in for the 'gdf' key's LSOA21CD column access resolve_stuck_areas needs."""
    def __init__(self, codes):
        self._codes = codes

    def __getitem__(self, key):
        assert key == "LSOA21CD"
        return self

    def tolist(self):
        return self._codes


class TestAdjustedDiagnosticsReport:
    def test_counts_match_classification(self, classifier_trace):
        report = adjusted_diagnostics_report(classifier_trace, "lag_P_lambda_weights", "P_like")
        assert report["n_flagged"] == 5
        assert report["n_hard_genuine"] == 1
        assert report["n_stuck_fixable"] == 1
        assert report["n_round_tripping"] == 1
        assert report["n_mixed"] == 1
        assert report["n_needs_review"] == 1
        assert report["n_needs_deep_dive"] == 2  # needs_review (1) + mixed (1), no resolution run

    def test_adjusted_rhat_excludes_hard_genuine_and_round_tripping(self, classifier_trace):
        report = adjusted_diagnostics_report(classifier_trace, "lag_P_lambda_weights", "P_like")
        # Every "locked" area in this fixture is similarly degenerate (near-infinite
        # r-hat by construction), so comparing magnitudes against the raw max isn't a
        # reliable test of the exclusion logic here — instead independently recompute
        # what the adjusted max SHOULD be (max r-hat over only stuck_fixable/mixed/
        # needs_review's positions) and check it matches exactly.
        rhat_da = az.rhat(classifier_trace, var_names=["lag_P_lambda_weights"])["lag_P_lambda_weights"]
        retained_positions = [IDX_STUCK_FIXABLE, IDX_MIXED, IDX_NEEDS_REVIEW]
        expected_adjusted_max = float(np.nanmax(rhat_da.values[retained_positions]))
        assert report["adjusted_max_rhat"] == pytest.approx(expected_adjusted_max)

    def test_within_chain_ess_reported_for_genuine_areas(self, classifier_trace):
        report = adjusted_diagnostics_report(classifier_trace, "lag_P_lambda_weights", "P_like")
        assert AREA_HARD_GENUINE in report["within_chain_ess"]
        assert AREA_ROUND_TRIPPING in report["within_chain_ess"]
        assert len(report["within_chain_ess"][AREA_HARD_GENUINE]) == N_CHAINS

    def test_best_case_excludes_every_flagged_area_not_just_two_categories(self, classifier_trace):
        # Every one of the fixture's 5 areas is flagged (one per category), so once
        # ALL of them are excluded there's nothing left to compute r-hat/ESS over —
        # unlike adjusted_max_rhat/min_ess, which still includes stuck_fixable/mixed/
        # needs_review's degenerate cells and so is never nan here.
        report = adjusted_diagnostics_report(classifier_trace, "lag_P_lambda_weights", "P_like")
        assert np.isnan(report["best_case_max_rhat"])
        assert np.isnan(report["best_case_min_ess"])
        assert not np.isnan(report["adjusted_max_rhat"])

    def test_best_case_keeps_only_never_flagged_areas(self):
        # 5 hand-built (always-flagged) areas plus one extra CLEAN area sampled iid
        # across chains (good r-hat/ESS by construction) — best_case should report
        # exactly the clean area's own r-hat/ESS, having thrown away every one of the
        # other 5 regardless of category.
        lw, ll = _build_classifier_arrays()
        rng = np.random.default_rng(99)
        clean_lw = rng.normal(0.5, 0.05, size=(N_CHAINS, N_DRAWS, 1, N_CATS))
        lw6 = np.concatenate([lw, clean_lw], axis=2)
        ll6 = np.concatenate([ll, np.zeros((N_CHAINS, N_DRAWS, 1, 1))], axis=2)
        codes6 = AREA_CODES + ["E99999999"]
        z_dummy = np.zeros((N_CHAINS, N_DRAWS, 6, 1))
        trace = az.from_dict(
            {"posterior": {"lag_P_lambda_weights": lw6, "z": z_dummy},
             "log_likelihood": {"P_like": ll6}},
            coords={"area": codes6}, dims={"z": ["area", "year"]})

        report = adjusted_diagnostics_report(trace, "lag_P_lambda_weights", "P_like")
        clean_area_rhat = az.rhat(
            trace, var_names=["lag_P_lambda_weights"])["lag_P_lambda_weights"].values[5]
        clean_area_ess = az.ess(
            trace, var_names=["lag_P_lambda_weights"], method="bulk"
        )["lag_P_lambda_weights"].values[5]
        assert report["best_case_max_rhat"] == pytest.approx(float(np.nanmax(clean_area_rhat)))
        assert report["best_case_min_ess"] == pytest.approx(float(np.nanmin(clean_area_ess)))
        # sanity: the clean area's own r-hat is nowhere near the flagged areas' —
        # confirms this is really reading the clean area, not accidentally still
        # picking up a flagged one.
        assert report["best_case_max_rhat"] < 1.05

    def test_resolved_verification_reduces_needs_deep_dive(self, classifier_trace):
        import pandas as pd
        # a stuck_fixable area that FAILED to resolve should count toward deep-dive
        fake_verification = pd.DataFrame(
            {"area": [AREA_STUCK_FIXABLE], "chain_agreement_frac": [0.5], "resolved": [False]})
        report = adjusted_diagnostics_report(
            classifier_trace, "lag_P_lambda_weights", "P_like",
            resolved_verification=fake_verification)
        assert report["n_needs_deep_dive"] == 3  # needs_review (1) + mixed (1) + unresolved stuck (1)


class TestAdjustedDiagnosticsSummary:
    """
    Unlike TestAdjustedDiagnosticsReport (single trace, caller-supplied loglik_var),
    adjusted_diagnostics_summary works over a dict of traces and auto-detects every
    `*_lambda_weights` var per trace — these tests exercise that multi-model wiring,
    the exclude_reviewed on/off split, the resolved_traces path, and the two
    passthrough edge cases (no lambda_weights var at all; a lambda_weights var with
    no matching log_likelihood entry to classify against). Every numeric assertion
    below was verified by actually running the function against this fixture first
    (same discipline as classifier_trace's own construction), not derived by hand.
    """

    def test_reports_multimodality_columns_default_exclude_reviewed(
            self, classifier_trace_with_sample_stats):
        diag = adjusted_diagnostics_summary(
            {"M": classifier_trace_with_sample_stats}, rhat_threshold=1.01)
        row = diag.loc["M"]
        assert row["n_lambda_weights_vars"] == 1
        assert row["n_flagged_multimodal"] == 5
        assert row["n_resolved"] == 0
        # mixed (1) + needs_review (1) + stuck_fixable, uncredited since no
        # resolved_traces was supplied for this model (1) — see the function's own
        # docstring: unlike adjusted_diagnostics_report, a never-attempted resolution
        # still counts as needing a deep-dive here, not as "known and tracked".
        assert row["n_needs_deep_dive"] == 3
        # exclude_reviewed=True (default) excludes all 5 areas (hard_genuine/
        # round_tripping always, mixed/needs_review/stuck_fixable because reviewed) —
        # nothing bad-r-hat is left in the aggregate.
        assert row["n_bad_rhat"] == 0
        assert row["min_ess"] == 12000
        # best_case is now scoped to *_lambda_weights cells UNION every scalar in this
        # model's diagnostic scope -- since no var_names dict was passed, that scope
        # falls back to every non-lag-var posterior variable, i.e. the dummy 'z' here.
        # Every one of this fixture's 5 areas is flagged (best_case excludes them all),
        # leaving only 'z' -- its rhat is NaN (degenerate constant series, never
        # contributes to max), but its ess is a real finite number (12000), so
        # best_case_min_ess picks that up even though best_case_max_rhat stays NaN.
        assert np.isnan(row["best_case_max_rhat"])
        assert row["best_case_min_ess"] == 12000
        assert row["min_ess"] == 12000

    def test_exclude_reviewed_false_keeps_reviewed_areas_in_aggregate(
            self, classifier_trace_with_sample_stats):
        diag = adjusted_diagnostics_summary(
            {"M": classifier_trace_with_sample_stats}, rhat_threshold=1.01,
            exclude_reviewed=False)
        row = diag.loc["M"]
        # stuck_fixable/mixed/needs_review are no longer excluded (only hard_genuine/
        # round_tripping are) -- their degenerate, still-unproven-benign r-hat now
        # shows up in the aggregate again.
        assert row["n_bad_rhat"] == 7
        assert row["min_ess"] == 8
        # status counts don't depend on exclude_reviewed -- it only changes which
        # cells feed the aggregate r-hat/ESS, not what's flagged/needing review.
        assert row["n_flagged_multimodal"] == 5
        assert row["n_needs_deep_dive"] == 3
        # best_case is UNAFFECTED by exclude_reviewed=False -- it always excludes every
        # flagged area regardless, so it matches the previous (exclude_reviewed=True)
        # test's own best_case exactly: NaN max_rhat (every lambda_weights area
        # excluded, 'z' contributes no finite rhat), 12000 min_ess (from 'z').
        assert np.isnan(row["best_case_max_rhat"])
        assert row["best_case_min_ess"] == 12000
        assert not np.isnan(row["max_rhat"])

    def test_resolved_traces_credits_resolved_stuck_area(
            self, classifier_trace_with_sample_stats):
        # Standing in as its own "resolved" run: the stuck_fixable area's chains
        # already agree 7/8 (87.5%), comfortably above verify_resolution's default
        # 0.75 purity_threshold, so this exercises the resolved_traces wiring
        # end-to-end without needing a second hand-built trace.
        resolved_traces = {"M": classifier_trace_with_sample_stats}
        diag = adjusted_diagnostics_summary(
            {"M": classifier_trace_with_sample_stats}, resolved_traces=resolved_traces,
            rhat_threshold=1.01, exclude_reviewed=True)
        row = diag.loc["M"]
        assert row["n_resolved"] == 1
        # needs_review (1) + mixed (1); stuck_fixable no longer counts, now resolved
        assert row["n_needs_deep_dive"] == 2
        assert row["n_bad_rhat"] == 0
        assert row["min_ess"] == 12000

        diag_conservative = adjusted_diagnostics_summary(
            {"M": classifier_trace_with_sample_stats}, resolved_traces=resolved_traces,
            rhat_threshold=1.01, exclude_reviewed=False)
        row_c = diag_conservative.loc["M"]
        # exclude_reviewed=False now maps directly to adjusted_diagnostics_report's own
        # "adjusted" semantics (hard_genuine/round_tripping only) -- it no longer
        # separately credits a confirmed-resolved stuck_fixable area the way the old
        # implementation did, so the resolved area's 2 category cells are back in the
        # aggregate here (5 -> 7) even though n_resolved (above) correctly shows it
        # resolved. Use exclude_reviewed=True (best_case) for a reading that credits it.
        assert row_c["n_bad_rhat"] == 7
        assert row_c["min_ess"] == 8

    def test_no_lambda_weights_var_matches_plain_diagnostics_summary(self):
        rng = np.random.default_rng(3)
        n_chains, n_draws = 4, 200
        sigma = np.abs(rng.normal(5.0, 0.1, size=(n_chains, n_draws)))
        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        trace = az.from_dict(
            {"posterior": {"sigma_plan": sigma}, "sample_stats": {"diverging": diverging}})

        plain    = diagnostics_summary({"M": trace})
        adjusted = adjusted_diagnostics_summary({"M": trace})

        assert adjusted.loc["M", "max_rhat"] == pytest.approx(plain.loc["M", "max_rhat"])
        assert adjusted.loc["M", "min_ess"] == plain.loc["M", "min_ess"]
        assert adjusted.loc["M", "n_lambda_weights_vars"] == 0
        assert adjusted.loc["M", "n_flagged_multimodal"] == 0
        assert adjusted.loc["M", "n_needs_deep_dive"] == 0

    def test_lambda_weights_without_log_likelihood_group_reported_unadjusted(self):
        # A lambda_weights var can exist with no log_likelihood group at all (not just
        # a missing/mismatched var within it) -- accessing trace.log_likelihood then
        # raises AttributeError rather than behaving like an empty/absent mapping, so
        # this exercises that guard specifically, not just the missing-var case.
        rng = np.random.default_rng(5)
        n_chains, n_draws, n_areas, n_cats = 4, 200, 3, 2
        lw = rng.random(size=(n_chains, n_draws, n_areas, n_cats))
        lw = lw / lw.sum(axis=-1, keepdims=True)
        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        trace = az.from_dict(
            {"posterior": {"lag_Q_lambda_weights": lw},
             "sample_stats": {"diverging": diverging}})

        plain    = diagnostics_summary({"M": trace})
        adjusted = adjusted_diagnostics_summary({"M": trace})

        # nothing could be classified, so the aggregate is reported unadjusted --
        # matching plain diagnostics_summary exactly, not dropped from the aggregate.
        assert adjusted.loc["M", "max_rhat"] == pytest.approx(plain.loc["M", "max_rhat"])
        assert adjusted.loc["M", "min_ess"] == plain.loc["M", "min_ess"]
        assert adjusted.loc["M", "n_lambda_weights_vars"] == 1
        assert adjusted.loc["M", "n_flagged_multimodal"] == 0
        assert adjusted.loc["M", "n_needs_deep_dive"] == 0

    def test_scalar_stuck_fixable_excluded_from_best_case_not_from_adjusted(self):
        # a flagged scalar (decisive gap, stuck_fixable) passed via var_names should be
        # excluded from best_case (every category except not_multimodal is excluded there)
        # but NOT from adjusted (only hard_genuine/round_tripping are excluded there) --
        # exercises that classify_scalar_multimodality's findings feed the SAME
        # exclusion machinery as lag-var findings, not a separate, inconsistent path.
        rng = np.random.default_rng(12)
        n_chains, n_draws = 8, 400
        scalar_vals = np.stack([rng.normal(mu, 1.0, n_draws) for mu in [0] * 6 + [8] * 2])
        logp = np.stack([rng.normal(-40.0, 0.5, n_draws) for _ in range(6)]
                        + [rng.normal(-70.0, 0.5, n_draws) for _ in range(2)])
        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        trace = az.from_dict(
            {"posterior": {"my_scalar": scalar_vals},
             "sample_stats": {"logp": logp, "diverging": diverging}})

        diag = adjusted_diagnostics_summary(
            {"M": trace}, var_names={"M": ["my_scalar"]}, rhat_threshold=1.01)
        row = diag.loc["M"]
        assert row["n_flagged_multimodal"] == 1
        assert row["n_not_multimodal"] == 0
        # best_case (exclude_reviewed=True, the default) excludes the flagged scalar cell
        # entirely -- nothing else in this trace, so nothing finite is left.
        assert np.isnan(row["best_case_max_rhat"])

        diag_conservative = adjusted_diagnostics_summary(
            {"M": trace}, var_names={"M": ["my_scalar"]}, rhat_threshold=1.01,
            exclude_reviewed=False)
        row_c = diag_conservative.loc["M"]
        # adjusted only excludes hard_genuine/round_tripping -- stuck_fixable stays in.
        assert not np.isnan(row_c["max_rhat"])
        assert row_c["n_bad_rhat"] >= 1

    def test_scalar_not_multimodal_never_excluded_from_best_case(self):
        # a flagged scalar with NO chain-level clustering (not_multimodal) must survive
        # even in best_case -- excluding it would misrepresent a real, unexplained problem
        # as "already accounted for by multimodality triage".
        rng = np.random.default_rng(13)
        n_chains, n_draws = 8, 400
        scalar_vals = np.stack(
            [rng.normal(mu, 1.0, n_draws) for mu in np.linspace(-3, 3, n_chains)])
        logp = rng.normal(-50.0, 0.5, size=(n_chains, n_draws))
        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        trace = az.from_dict(
            {"posterior": {"my_scalar": scalar_vals},
             "sample_stats": {"logp": logp, "diverging": diverging}})

        diag = adjusted_diagnostics_summary(
            {"M": trace}, var_names={"M": ["my_scalar"]}, rhat_threshold=1.01)
        row = diag.loc["M"]
        assert row["n_not_multimodal"] == 1
        # best_case must NOT exclude it -- its bad r-hat is still visible.
        assert not np.isnan(row["best_case_max_rhat"])
        assert row["best_case_max_rhat"] == pytest.approx(row["max_rhat"])


@pytest.fixture(scope="module")
def two_lag_var_trace():
    """classifier_trace's five areas, duplicated under BOTH lag_P_lambda_weights/P_like AND
    lag_E_lambda_weights/E_like -- same underlying numbers on both, deliberately. This
    fixture exists to exercise multimodality_report's cross-lag-var COMBINING logic (does
    it sum counts across both vars, take max/min correctly, tag classification_df/
    within_chain_ess by lag_var), not to give P and E a different profile -- the real
    AZ4b worked example in docs/multimodality-diagnostic-pipeline.md already demonstrates
    that P and E can diverge sharply in practice."""
    lw, ll = _build_classifier_arrays()
    z_dummy = np.zeros((N_CHAINS, N_DRAWS, N_AREAS, 1))
    return az.from_dict(
        {"posterior": {"lag_P_lambda_weights": lw, "lag_E_lambda_weights": lw.copy(), "z": z_dummy},
         "log_likelihood": {"P_like": ll, "E_like": ll.copy()}},
        coords={"area": AREA_CODES},
        dims={"z": ["area", "year"]},
    )


class TestMultimodalityReport:
    """multimodality_report is the function check-multimodality actually calls -- it exists
    specifically to make check-multimodality and adjusted_diagnostics_summary (diagnose
    --adjust-for-multimodality) agree on raw/adjusted/best_case for the same trace, by
    checking every *_lambda_weights var a trace has instead of just one (see the function's
    own docstring for the AZ4b incident this fixes)."""

    def test_auto_detects_both_lag_vars(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        assert set(report["lag_vars_checked"]) == {"lag_P_lambda_weights", "lag_E_lambda_weights"}
        assert report["lag_vars_skipped"] == []

    def test_counts_sum_across_lag_vars(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        # each lag var independently flags the same 5 areas (identical underlying data) --
        # combined counts double the single-lag-var picture from TestAdjustedDiagnosticsReport.
        assert report["n_flagged"] == 10
        assert report["n_hard_genuine"] == 2
        assert report["n_stuck_fixable"] == 2
        assert report["n_round_tripping"] == 2
        assert report["n_mixed"] == 2
        assert report["n_needs_review"] == 2
        assert report["n_needs_deep_dive"] == 4  # (needs_review(1) + mixed(1), no resolution) x2

    def test_raw_and_adjusted_combine_across_lag_vars(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        single = adjusted_diagnostics_report(
            two_lag_var_trace, "lag_P_lambda_weights", "P_like")
        # identical data on both lag vars -> combined max/min equal the single-var reading,
        # not double-counted or otherwise distorted by combining two identical inputs.
        assert report["raw_max_rhat"] == pytest.approx(single["raw_max_rhat"])
        assert report["raw_min_ess"] == pytest.approx(single["raw_min_ess"])
        assert report["adjusted_max_rhat"] == pytest.approx(single["adjusted_max_rhat"])
        assert report["adjusted_min_ess"] == pytest.approx(single["adjusted_min_ess"])

    def test_best_case_nan_when_every_area_flagged_on_every_lag_var(self, two_lag_var_trace):
        # every one of this fixture's 5 areas is flagged on BOTH lag vars, so best_case is
        # NaN on each individually -- combining two all-NaN readings must stay NaN, not
        # silently become 0 or raise on an empty max/min reduction.
        report = multimodality_report(two_lag_var_trace)
        assert np.isnan(report["best_case_max_rhat"])
        assert np.isnan(report["best_case_min_ess"])

    def test_classification_df_tagged_by_lag_var(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        df = report["classification_df"]
        assert len(df) == 10
        assert set(df["lag_var"]) == {"lag_P_lambda_weights", "lag_E_lambda_weights"}

    def test_within_chain_ess_keyed_by_lag_var_and_area(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        assert f"lag_P_lambda_weights:{AREA_HARD_GENUINE}" in report["within_chain_ess"]
        assert f"lag_E_lambda_weights:{AREA_HARD_GENUINE}" in report["within_chain_ess"]

    def test_by_lag_var_gives_each_lag_var_its_own_full_report(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace)
        assert set(report["by_lag_var"]) == {"lag_P_lambda_weights", "lag_E_lambda_weights"}
        assert report["by_lag_var"]["lag_P_lambda_weights"]["n_flagged"] == 5

    def test_lag_vars_param_restricts_to_explicit_subset(self, two_lag_var_trace):
        report = multimodality_report(two_lag_var_trace, lag_vars=["lag_P_lambda_weights"])
        assert report["lag_vars_checked"] == ["lag_P_lambda_weights"]
        assert report["n_flagged"] == 5

    def test_resolved_trace_verifies_per_lag_var(self, two_lag_var_trace):
        # standing in as "resolved": the stuck_fixable area's chains already agree 7/8 on
        # BOTH lag vars here (P and E share identical underlying data), comfortably above
        # verify_resolution's default 0.75 threshold -- exercises that resolution is
        # checked independently per lag var, not just once against whichever lag var
        # happens to be first.
        report = multimodality_report(two_lag_var_trace, resolved_trace=two_lag_var_trace)
        assert report["n_resolved"] == 2
        assert report["n_needs_deep_dive"] == 4  # (needs_review + mixed) x2, stuck now credited

    def test_lag_var_without_log_likelihood_group_is_skipped(self):
        rng = np.random.default_rng(5)
        n_chains, n_draws, n_areas, n_cats = 4, 200, 3, 2
        lw = rng.random(size=(n_chains, n_draws, n_areas, n_cats))
        lw = lw / lw.sum(axis=-1, keepdims=True)
        trace = az.from_dict({"posterior": {"lag_Q_lambda_weights": lw}})

        report = multimodality_report(trace)
        assert report["lag_vars_checked"] == []
        assert report["lag_vars_skipped"] == ["lag_Q_lambda_weights"]
        assert report["n_flagged"] == 0
        # a skipped lag var can't be CLASSIFIED, but its cells still belong in the model's
        # diagnostic scope -- raw/adjusted/best_case all still see them unfiltered (matching
        # plain diagnose's own reading), rather than silently dropping them from the
        # aggregate entirely.
        expected_max_rhat = float(
            az.rhat(trace, var_names=["lag_Q_lambda_weights"])["lag_Q_lambda_weights"]
            .values.max())
        assert report["raw_max_rhat"] == pytest.approx(expected_max_rhat)
        assert report["adjusted_max_rhat"] == pytest.approx(expected_max_rhat)
        assert report["best_case_max_rhat"] == pytest.approx(expected_max_rhat)

    def test_no_lag_vars_at_all(self):
        trace = az.from_dict({"posterior": {"sigma_plan": np.ones((2, 10))}})
        report = multimodality_report(trace)
        assert report["lag_vars_checked"] == []
        assert report["lag_vars_skipped"] == []
        assert report["n_flagged"] == 0

    def test_combines_lag_var_and_scalar_findings(self, classifier_trace):
        # classifier_trace already has one flagged area per category on
        # lag_P_lambda_weights; add ONE flagged scalar with a decisive gap (stuck_fixable)
        # onto the same trace's posterior/sample_stats, and confirm multimodality_report
        # folds it into the same combined counts/raw/adjusted/best_case, over the SAME
        # combined scope raw/best_case both draw from.
        rng = np.random.default_rng(11)
        n_draws = N_DRAWS
        scalar_vals = np.stack([rng.normal(mu, 1.0, n_draws) for mu in [0] * 6 + [8] * 2])
        scalar_logp = np.stack([rng.normal(-40.0, 0.5, n_draws) for _ in range(6)]
                               + [rng.normal(-70.0, 0.5, n_draws) for _ in range(2)])

        posterior = {k: v.values for k, v in classifier_trace.posterior.data_vars.items()}
        posterior["my_scalar"] = scalar_vals
        log_likelihood = {
            k: v.values for k, v in classifier_trace.log_likelihood.data_vars.items()}
        trace = az.from_dict(
            {"posterior": posterior, "log_likelihood": log_likelihood,
             "sample_stats": {"logp": scalar_logp}},
            coords={"area": AREA_CODES}, dims={"z": ["area", "year"]})

        report = multimodality_report(trace, var_names=["my_scalar"])

        # 5 lag-var findings (unchanged from the single-lag-var report) + 1 scalar finding
        assert report["n_flagged"] == 6
        # classifier_trace's own stuck_fixable area (1) + the new scalar's stuck_fixable (1)
        assert report["n_stuck_fixable"] == 2
        assert not report["scalar_classification_df"].empty
        assert report["scalar_classification_df"].iloc[0]["var"] == "my_scalar"

        # best_case excludes the scalar's own cell (stuck_fixable is always excluded from
        # best_case) -- combining it must not silently drop the lag-var exclusions already
        # verified in TestAdjustedDiagnosticsReport, nor vice versa. Since every one of
        # this trace's 5 areas is ALSO flagged (same construction as classifier_trace) and
        # the scalar's own cell is excluded too, nothing should remain finite.
        assert np.isnan(report["best_case_max_rhat"])

    def test_lag_vars_empty_list_vs_none_are_equivalent_when_none_exist(self):
        # passing lag_vars=[] explicitly must behave the same as omitting it (None) when
        # the trace genuinely has no *_lambda_weights var -- both should just skip the
        # lag-var side entirely, not raise or behave differently.
        trace = az.from_dict({"posterior": {"sigma_plan": np.ones((2, 10))}})
        report_default = multimodality_report(trace)
        report_explicit_empty = multimodality_report(trace, lag_vars=[])
        assert report_default["lag_vars_checked"] == report_explicit_empty["lag_vars_checked"]
        assert report_default["n_flagged"] == report_explicit_empty["n_flagged"]


@pytest.mark.slow
class TestResolveStuckAreasIntegration:
    """
    End-to-end plumbing check: does resolve_stuck_areas actually drive a real
    DwellingModel's sample() call with a real init_mean and come back with a valid
    trace? Uses AZ1d on the tiny 9-area synthetic grid (conftest's data_dict) with a
    manually-constructed classification_df — not asserting the tiny fixture's data
    produces any particular real ambiguity (it's far too small/fast a run for that),
    just that the wiring (seed construction -> model.sample(init_mean=...) ->
    verify_resolution) runs without error and returns well-formed output.
    """

    def test_end_to_end_resample_and_verify(self, data_dict):
        import pandas as pd

        from housing_projections.models.models import AZ1d

        # pretend area 0 is stuck_fixable, favouring category 1 -- whether or not
        # that's actually true for this tiny synthetic dataset is irrelevant to what
        # this test checks (the plumbing, not the science).
        area_code = data_dict["gdf"]["LSOA21CD"].iloc[0]
        classification_df = pd.DataFrame([{
            "area": area_code, "max_rhat": 1.5, "mean_switch_rate": 0.0,
            "n_pure_chains": 4, "n_mixed_chains": 0, "best_category": 1,
            "gaps": {0: 20.0}, "category": "stuck_fixable",
            "recommended_action": "reseed via resolve_stuck_areas and resample",
        }])

        trace, seeded = resolve_stuck_areas(
            AZ1d, data_dict, classification_df, "lag_P_lambda_weights",
            chains=2, draws=15, tune=15, target_accept=0.8, random_seed=0)

        assert seeded == [area_code]
        assert "lag_P_lambda_weights" in trace.posterior
        assert "P_like" in trace.log_likelihood

        result = verify_resolution(trace, "lag_P_lambda_weights", seeded)
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]["chain_agreement_frac"] <= 1.0
