# %%
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from housing_projections.load import get_dwellings

dwellings_all, year_cols_ben, year_cols_completion = get_dwellings()

dwellings_all = dwellings_all[dwellings_all['intercensal_completions'] < 2000]


# %%

# ── Data preparation ──────────────────────────────────────────────────────────

year_cols_planning = [f'{y}/{str(y+1)[-2:]}' for y in range(2009, 2025)]
year_cols_ben      = [f'{y}_ben'              for y in range(2009, 2025)]

# Years we want to infer: 2012 to 2021 inclusive (10 years)
# These correspond to net change in year ending in that year
infer_years        = list(range(2012, 2022))
infer_cols_plan    = [f'{y}/{str(y+1)[-2:]}' for y in range(2011, 2021)]
infer_cols_ben     = [f'{y}_ben'              for y in range(2011, 2021)]

n_areas = len(dwellings_all)
n_years = len(infer_years)   # 10

# Census totals — the hard constraint
C_2011  = dwellings_all['dwellings_2011'].values.astype(float)  # (n_areas,)
C_2021  = dwellings_all['dwellings_2021'].values.astype(float)
D       = C_2021 - C_2011                                        # (n_areas,) known total change

# Observed noisy measurements — shape (n_areas, n_years)
P_obs   = dwellings_all[infer_cols_plan].values.astype(float)
E_obs   = dwellings_all[infer_cols_ben].values.astype(float)

n_test = 80
dwellings_test = dwellings_all.iloc[:n_test]
D_test  = D[:n_test]
P_test  = P_obs[:n_test]
E_test  = E_obs[:n_test]

# ── Reparameterisation to enforce census constraint ───────────────────────────

def build_z(z_devs, D, n_years):
    """
    For each area i:
      - baseline is D[i] / n_years  (uniform allocation of that area's census diff)
      - deviations are centred per area so they sum to zero across years
      - result sums exactly to D[i] for every area individually
    """
    # Uniform baseline per area — shape (n_areas, 1)
    baseline = D[:, None] / n_years

    # Centre deviations per area along year axis
    # Each area's deviations sum to zero independently
    z_devs_centred = z_devs - z_devs.mean(axis=1, keepdims=True)

    return baseline + z_devs_centred


# ── M0: baseline — both sources unbiased, equal variance ─────────────────────

with pm.Model() as M0:

    # Global mean annual change — informs the prior on deviations
    # but the hard per-area constraint comes from baseline = D[i] / n_years
    sigma_z = pm.HalfNormal('sigma_z', sigma=50)

    # All n_years treated symmetrically, shape (n_areas, n_years)
    z_devs = pm.Normal('z_devs',
                       mu=0,
                       sigma=sigma_z,
                       shape=(n_areas, n_years))

    # Constraint enforced per area, symmetrically across years
    z = pm.Deterministic('z', build_z(z_devs, D, n_years))

    sigma_obs = pm.HalfNormal('sigma_obs', sigma=30)

    P_like = pm.Normal('P_like', mu=z, sigma=sigma_obs, observed=P_obs)
    E_like = pm.Normal('E_like', mu=z, sigma=sigma_obs, observed=E_obs)

    prior = pm.sample_prior_predictive(draws=2000)


# %% Prior predictive checks 

# ── 1. Check the census constraint is exactly satisfied ───────────────────────
# For every draw and every LSOA, z should sum to D
z_prior = prior.prior['z'].values   # shape (chain, draw, n_areas, n_years)
z_sums  = z_prior.sum(axis=-1)      # sum over years → (chain, draw, n_areas)
residuals = z_sums - D[None, None, :]

print(f"Max constraint violation: {np.abs(residuals).max():.6f}")   # should be ~0

# ── 2. Distribution of implied annual changes ─────────────────────────────────
# Flatten across draws and areas to see the marginal prior on z
z_flat = z_prior.reshape(-1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(z_flat, bins=100, density=True, color='steelblue', alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Implied annual net dwelling change')
ax.set_ylabel('Density')
ax.set_title('Prior predictive: marginal distribution of z')
ax.set_xlim(-300, 300)   # zoom in to plausible range
plt.tight_layout()
plt.show()

# ── 3. Compare prior predictive observations to actual data ───────────────────
# P_like and E_like are the prior predictive observations
P_prior = prior.prior_predictive['P_like'].values.reshape(-1)
E_prior = prior.prior_predictive['E_like'].values.reshape(-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pred, obs, label in zip(
    axes,
    [P_prior, E_prior],
    [P_obs.ravel(), E_obs.ravel()],
    ['Planning completions', 'BEN estimates']
):
    ax.hist(pred, bins=100, density=True, alpha=0.5,
            color='steelblue', label='Prior predictive')
    ax.hist(obs,  bins=100, density=True, alpha=0.5,
            color='coral',     label='Observed')
    ax.set_xlabel('Net dwelling change')
    ax.set_ylabel('Density')
    ax.set_title(label)
    ax.set_xlim(-200, 200)
    ax.legend()

plt.suptitle('Prior predictive vs observed data')
plt.tight_layout()
plt.show()

# ── 4. Prior on sigma ─────────────────────────────────────────────────────────
sigma_prior = prior.prior['sigma_obs'].values.ravel()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(sigma_prior, bins=50, color='steelblue', alpha=0.7, density=True)
ax.axvline(P_obs.std(), color='coral',     linestyle='--', label='Observed SD (planning)')
ax.axvline(E_obs.std(), color='steelblue', linestyle='--', label='Observed SD (BEN)')
ax.set_xlabel('sigma_obs')
ax.set_title('Prior on observation noise vs empirical SD')
ax.legend()
plt.tight_layout()
plt.show()

# ── 5. Per-year prior predictive means vs observed means ─────────────────────
z_year_mean  = z_prior.mean(axis=(0, 1, 2))    # mean over chains, draws, areas
P_year_mean  = P_obs.mean(axis=0)
E_year_mean  = E_obs.mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_years, z_year_mean, marker='o', label='Prior z mean',       color='black')
ax.plot(infer_years, P_year_mean, marker='s', label='Planning obs mean',  color='steelblue')
ax.plot(infer_years, E_year_mean, marker='^', label='BEN obs mean',       color='coral')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean net dwelling change')
ax.set_title('Prior predictive means vs observed means by year')
ax.legend()
plt.tight_layout()
plt.show()
             
# %% Inference

# ── Sampling ──────────────────────────────────────────────────────────────────
# Start with a small run to check the model compiles and mixes
# n_areas=4987 with 9 free params each = ~45k parameters, so keep chains short initially

# Then in your sample call
SAMPLE_KWARGS = dict(
    draws=2000,
    tune=500,
    chains=4,
    cores=1,
    target_accept=0.95,
    random_seed=42,
    return_inferencedata=True,
)

with pm.Model() as M0_test:
    # Global mean annual change — informs the prior on deviations
    # but the hard per-area constraint comes from baseline = D[i] / n_years
    sigma_z = pm.HalfNormal('sigma_z', sigma=50)

    # All n_years treated symmetrically, shape (n_areas, n_years)
    z_devs = pm.Normal('z_devs',
                       mu=0,
                       sigma=sigma_z,
                       shape=(n_test, n_years))

    # Constraint enforced per area, symmetrically across years
    z = pm.Deterministic('z', build_z(z_devs, D_test, n_years))

    sigma_obs = pm.HalfNormal('sigma_obs', sigma=30)

    P_like = pm.Normal('P_like', mu=z, sigma=sigma_obs, observed=P_test)
    E_like = pm.Normal('E_like', mu=z, sigma=sigma_obs, observed=E_test)

    prior = pm.sample_prior_predictive(draws=2000)

    trace_test = pm.sample(draws=2000, tune=200, cores=1, chains=4)
    graph = pm.model_to_graphviz(M0_test)


# with M0:
#     trace_M0 = pm.sample(**SAMPLE_KWARGS)

# with M1:
#     trace_M1 = pm.sample(**SAMPLE_KWARGS)

# ── Comparison ────────────────────────────────────────────────────────────────

# print(az.summary(trace_test, var_names=['sigma_obs']))
# print(az.summary(trace_M1, var_names=['sigma_plan', 'sigma_ben']))

# Model comparison via LOO
# comparison = az.compare({'M0': trace_M0, 'M1': trace_M1})
# print(comparison)

# %%
# graph


# ── 1. Sample posterior predictive ───────────────────────────────────────────
with M0_test:
    post_pred = pm.sample_posterior_predictive(trace_test)

# ── 2. Check census constraint is satisfied in posterior ──────────────────────
z_post = trace_test.posterior['z'].values   # (chains, draws, n_areas, n_years)
z_sums = z_post.sum(axis=-1)              # sum over years → (chains, draws, n_areas)
residuals = z_sums - D_test[None, None, :]

print(f"Max per-area constraint violation: {np.abs(residuals).max():.6f}")

# ── 3. Posterior on global parameters ─────────────────────────────────────────
az.plot_trace(trace_test, var_names=['sigma_z', 'sigma_obs'])
plt.suptitle('M0 — posterior on global parameters')
plt.tight_layout()
plt.show()

# ── 4. Posterior predictive vs observed — planning and BEN ────────────────────
P_post = post_pred.posterior_predictive['P_like'].values   # (chains, draws, n_areas, n_years)
E_post = post_pred.posterior_predictive['E_like'].values

P_post_flat = P_post.reshape(-1)
E_post_flat = E_post.reshape(-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pred, obs, label in zip(
    axes,
    [P_post_flat, E_post_flat],
    [P_obs.ravel(), E_obs.ravel()],
    ['Planning completions', 'BEN estimates']
):
    ax.hist(pred, bins=100, density=True, alpha=0.5,
            color='steelblue', label='Posterior predictive')
    ax.hist(obs,  bins=100, density=True, alpha=0.5,
            color='coral',     label='Observed')
    ax.set_xlabel('Net dwelling change')
    ax.set_ylabel('Density')
    ax.set_title(label)
    ax.set_xlim(-200, 200)
    ax.legend()

plt.suptitle('Posterior predictive vs observed')
plt.tight_layout()
plt.show()

# ── 5. Posterior mean z by year vs observed means ─────────────────────────────
z_year_mean     = z_post.mean(axis=(0, 1, 2))          # mean over chains, draws, areas
z_year_lower    = np.percentile(z_post.mean(axis=2), 5,  axis=(0, 1))
z_year_upper    = np.percentile(z_post.mean(axis=2), 95, axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_years, z_year_mean, marker='o', color='black',
        label='Posterior mean z')
ax.fill_between(infer_years, z_year_lower, z_year_upper,
                alpha=0.2, color='black', label='90% CI on mean')
ax.plot(infer_years, P_obs.mean(axis=0), marker='s',
        color='steelblue', label='Planning obs mean')
ax.plot(infer_years, E_obs.mean(axis=0), marker='^',
        color='coral',     label='BEN obs mean')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean net dwelling change')
ax.set_title('Posterior mean z vs observed means by year')
ax.legend()
plt.tight_layout()
plt.show()

# ── 6. Posterior z for a sample of individual areas ───────────────────────────
n_sample = 6
sample_idx = np.random.choice(n_test, n_sample, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
for ax, idx in zip(axes.ravel(), sample_idx):
    z_area = z_post[:, :, idx, :]          # (chains, draws, n_years)
    z_mean = z_area.mean(axis=(0, 1))
    z_lo   = np.percentile(z_area, 5,  axis=(0, 1))
    z_hi   = np.percentile(z_area, 95, axis=(0, 1))

    ax.plot(infer_years, z_mean, color='black', marker='o', label='Posterior mean')
    ax.fill_between(infer_years, z_lo, z_hi, alpha=0.2, color='black', label='90% CI')
    ax.plot(infer_years, P_obs[idx], color='steelblue',
            marker='s', alpha=0.7, label='Planning')
    ax.plot(infer_years, E_obs[idx], color='coral',
            marker='^', alpha=0.7, label='BEN')
    ax.axhline(0, color='black', linewidth=0.5)
    # Uniform baseline — D[i] / n_years
    ax.axhline(D[idx] / n_years, color='green', linewidth=0.8,
           linestyle='--', alpha=0.5, label=f'D/n ({D[idx]/n_years:.1f})')
    ax.set_title(f'LSOA {idx}  (census diff={D[idx]:.0f})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net change')
    ax.legend(fontsize=7)

plt.suptitle('Posterior z for sampled LSOAs')
plt.tight_layout()
plt.show()

# ── 7. Distribution of posterior uncertainty (width of CI) across areas ───────
z_lo_all = np.percentile(z_post, 5,  axis=(0, 1))   # (n_areas, n_years)
z_hi_all = np.percentile(z_post, 95, axis=(0, 1))
ci_width = z_hi_all - z_lo_all                        # (n_areas, n_years)

fig, ax = plt.subplots(figsize=(10, 4))
ax.boxplot([ci_width[:, t] for t in range(n_years)],
           positions=infer_years, widths=0.6, showfliers=False)
ax.set_xlabel('Year')
ax.set_ylabel('90% CI width')
ax.set_title('Posterior uncertainty width by year across all LSOAs')
plt.tight_layout()
plt.show()
# %%
