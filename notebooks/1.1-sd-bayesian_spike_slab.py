# %% [markdown]
# > **ARCHIVED** — This notebook uses the old `get_dwellings()` API and will
# > not run. Kept for historical reference. See `1.2-sd-inference.py` for the
# > current workflow.

# %%
import pymc as pm
import pytensor.tensor as pt
from housing_projections.data import get_dwellings
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pandas as pd

dwellings_all, year_cols_ben, year_cols_completion = get_dwellings()

dwellings_all = dwellings_all[dwellings_all['intercensal_completions'] < 2000]

# %%

# ── Data preparation ──────────────────────────────────────────────────────────
n_areas = len(dwellings_all)
n_years = 10

year_cols_planning = [f'{y}/{str(y+1)[-2:]}' for y in range(2009, 2025)]
year_cols_ben      = [f'{y}_ben'              for y in range(2009, 2025)]

# Years we want to infer: 2012 to 2021 inclusive (10 years)
# These correspond to net change in year ending in that year
infer_years        = list(range(2012, 2022))
infer_cols_plan    = [f'{y}/{str(y+1)[-2:]}' for y in range(2011, 2021)]
infer_cols_ben     = [f'{y}_ben'              for y in range(2011, 2021)]

C_2011  = dwellings_all['dwellings_2011'].values.astype(float)
C_2021  = dwellings_all['dwellings_2021'].values.astype(float)
D       = (C_2021 - C_2011).astype(float)

P_obs   = dwellings_all[infer_cols_plan].values.astype(float)
E_obs   = dwellings_all[infer_cols_ben].values.astype(float)

# ── Helper: marginalised spike-and-slab log probability ───────────────────────
def spike_slab_logp(z, pi, mu_slab, sigma_spike, sigma_slab, nu):
    """
    Two-component spike-and-slab:
      - spike: Normal(0, sigma_spike)              with prob pi
      - slab:  StudentT(nu, mu_slab, sigma_slab)   with prob (1-pi)
    """
    log_p_spike = (
        pt.log(pi)
        + pm.logp(pm.Normal.dist(mu=0, sigma=sigma_spike), z)
    )
    log_p_slab  = (
        pt.log(1 - pi)
        + pm.logp(pm.StudentT.dist(nu=nu, mu=mu_slab, sigma=sigma_slab), z)
    )

    return pt.logsumexp(
        pt.stack([log_p_spike, log_p_slab], axis=0), axis=0
    )


# ── Sampling configuration — change these ─────────────────────────────────────
N_AREAS   = 150     # set to n_areas for full run
N_DRAWS   = 1000
N_TUNE    = 500
N_CHAINS  = 2
N_CORES   = 1
TARGET_ACCEPT = 0.9
RANDOM_SEED   = 42

# ── Subset data ───────────────────────────────────────────────────────────────
D_sub   = D[:N_AREAS]
P_sub   = P_obs[:N_AREAS]
E_sub   = E_obs[:N_AREAS]

# ── Build and sample model ────────────────────────────────────────────────────
CENSUS_REL_ERROR = 0.02
CENSUS_ABS_FLOOR = 2.0

with pm.Model() as M0:

    # # Global mixture parameters
    # pi          = pm.Beta('pi',       alpha=4.5, beta=5.5)
    # mu_slab     = pm.Normal('mu_slab', mu=D.mean() / n_years / 0.55, sigma=5)
    # sigma_slab  = pm.HalfNormal('sigma_slab',  sigma=30)
    # nu          = pm.Gamma('nu',      alpha=2,   beta=0.1)
    # sigma_obs   = pm.HalfNormal('sigma_obs',   sigma=30)

    # # Latent true changes
    # z = pm.Normal('z',
    #               mu=D_sub[:, None] / n_years,
    #               sigma=20,
    #               shape=(N_AREAS, n_years))

    # # Census constraint
    # pm.Normal('census_obs', mu=z.sum(axis=1), sigma=0.5, observed=D_sub)

    # # Spike-and-slab prior
    # logp_z = spike_slab_logp(z, pi, mu_slab, sigma_spike=1.0,
    #                           sigma_slab=sigma_slab, nu=nu)
    # pm.Potential('spike_slab_prior', logp_z.sum())

    # # Likelihoods
    # pm.Normal('P_like', mu=z, sigma=sigma_obs, observed=P_sub)
    # pm.Normal('E_like', mu=z, sigma=sigma_obs, observed=E_sub)

    mu_slab    = pm.Normal('mu_slab', mu=D.mean() / n_years / 0.55, sigma=5)
    sigma_slab = pm.HalfNormal('sigma_slab', sigma=30)
    sigma_obs  = pm.HalfNormal('sigma_obs',  sigma=30)

    # Simple Normal prior instead of spike-and-slab
    z = pm.Normal('z',
                  mu=mu_slab,
                  sigma=sigma_slab,
                  shape=(N_AREAS, n_years))

    # pm.Normal('census_obs', mu=z.sum(axis=1), sigma=0.5, observed=D_sub)
    sigma_census = np.maximum(np.abs(D_sub) * CENSUS_REL_ERROR, CENSUS_ABS_FLOOR)
    pm.Normal('census_obs', mu=z.sum(axis=1), sigma=sigma_census, observed=D_sub)

    pm.Normal('P_like', mu=z, sigma=sigma_obs, observed=P_sub)
    pm.Normal('E_like', mu=z, sigma=sigma_obs, observed=E_sub)

    
# ── Prior predictive check ────────────────────────────────────────────────────
with M0:
    prior = pm.sample_prior_predictive(draws=200)
    graph = pm.model_to_graphviz(M0)

graph

# %%
# z_prior      = prior.prior['z'].values              # (chains, draws, n_areas, n_years)
# pi_prior     = prior.prior['pi'].values.ravel()
# mu_slab_prior   = prior.prior['mu_slab'].values.ravel()
# sigma_slab_prior = prior.prior['sigma_slab'].values.ravel()
# nu_prior     = prior.prior['nu'].values.ravel()
# sigma_obs_prior  = prior.prior['sigma_obs'].values.ravel()

# P_prior      = prior.prior_predictive['P_like'].values
# E_prior      = prior.prior_predictive['E_like'].values

# fig, axes = plt.subplots(2, 4, figsize=(20, 8))

# # ── 1. Marginal distribution of z ─────────────────────────────────────────────
# ax = axes[0, 0]
# z_flat = z_prior.reshape(-1)
# ax.hist(z_flat, bins=200, density=True, color='steelblue', alpha=0.7)
# ax.axvline(0, color='black', linewidth=0.8)
# ax.set_xlim(-150, 150)
# ax.set_xlabel('z')
# ax.set_title('Prior: marginal distribution of z')
# frac_near_zero = np.mean(np.abs(z_flat) < 3)
# ax.text(0.05, 0.95, f'P(|z|<3) = {frac_near_zero:.2f}',
#         transform=ax.transAxes, verticalalignment='top', fontsize=9)

# # ── 2. Prior predictive vs observed: planning ─────────────────────────────────
# ax = axes[0, 1]
# ax.hist(P_prior.reshape(-1), bins=200, density=True,
#         alpha=0.5, color='steelblue', label='Prior predictive')
# ax.hist(P_obs.ravel(), bins=200, density=True,
#         alpha=0.5, color='coral', label='Observed')
# ax.set_xlim(-150, 150)
# ax.set_xlabel('Net dwelling change')
# ax.set_title('Prior predictive vs observed: planning')
# ax.legend()

# # ── 3. Prior predictive vs observed: BEN ──────────────────────────────────────
# ax = axes[0, 2]
# ax.hist(E_prior.reshape(-1), bins=200, density=True,
#         alpha=0.5, color='steelblue', label='Prior predictive')
# ax.hist(E_obs.ravel(), bins=200, density=True,
#         alpha=0.5, color='coral', label='Observed')
# ax.set_xlim(-150, 150)
# ax.set_xlabel('Net dwelling change')
# ax.set_title('Prior predictive vs observed: BEN')
# ax.legend()

# # ── 4. pi ─────────────────────────────────────────────────────────────────────
# ax = axes[0, 3]
# ax.hist(pi_prior, bins=50, density=True, alpha=0.7, color='steelblue')
# ax.axvline(0.45, color='black', linestyle='--', linewidth=0.8,
#            label='prior centre (0.45)')
# ax.set_xlabel('pi')
# ax.set_ylabel('Density')
# ax.set_title('Prior: pi (spike probability)')
# ax.legend()

# # ── 5. mu_slab ────────────────────────────────────────────────────────────────
# ax = axes[1, 0]
# ax.hist(mu_slab_prior, bins=50, density=True, alpha=0.7, color='coral')
# ax.axvline(D.mean() / n_years / 0.55, color='black', linestyle='--',
#            linewidth=0.8, label=f'prior centre ({D.mean()/n_years/0.55:.1f})')
# ax.set_xlabel('mu_slab')
# ax.set_ylabel('Density')
# ax.set_title('Prior: mu_slab (mean change in active LSOA-years)')
# ax.legend()

# # ── 6. nu ─────────────────────────────────────────────────────────────────────
# ax = axes[1, 1]
# ax.hist(nu_prior, bins=50, density=True, color='steelblue', alpha=0.7)
# ax.axvline(4,  color='black', linestyle='--', linewidth=0.8, label='nu=4 (heavy)')
# ax.axvline(30, color='red',   linestyle='--', linewidth=0.8, label='nu=30 (normal-like)')
# ax.set_xlabel('nu')
# ax.set_title('Prior: degrees of freedom')
# ax.legend()

# # ── 7. Scale parameters ───────────────────────────────────────────────────────
# ax = axes[1, 2]
# ax.hist(sigma_slab_prior, bins=50, density=True, alpha=0.7,
#         color='steelblue', label='sigma_slab')
# ax.hist(sigma_obs_prior,  bins=50, density=True, alpha=0.7,
#         color='coral',     label='sigma_obs')
# ax.axvline(P_obs.std(), color='steelblue', linestyle='--',
#            linewidth=0.8, label='observed SD (planning)')
# ax.axvline(E_obs.std(), color='coral',     linestyle='--',
#            linewidth=0.8, label='observed SD (BEN)')
# ax.set_xlabel('Value')
# ax.set_title('Prior: scale parameters vs empirical SD')
# ax.legend(fontsize=7)

# # ── 8. Census constraint check ────────────────────────────────────────────────
# ax = axes[1, 3]
# z_sums    = z_prior.sum(axis=-1).reshape(-1, n_areas)
# residuals = (z_sums - D[None, :]).ravel()
# ax.hist(residuals, bins=100, density=True, color='steelblue', alpha=0.7)
# ax.axvline(0, color='black', linewidth=0.8)
# ax.set_xlabel('z sum - D')
# ax.set_title('Census constraint violations (prior)')
# ax.text(0.05, 0.95,
#         f'mean={np.abs(residuals).mean():.2f}\nmax={np.abs(residuals).max():.2f}',
#         transform=ax.transAxes, verticalalignment='top', fontsize=9)

# plt.suptitle('Prior predictive checks — M0 (spike and slab)')
# plt.tight_layout()
# plt.show()


# %% Sampling

# ── Sampling configuration — change these ─────────────────────────────────────
N_AREAS   = 150     # set to n_areas for full run
N_DRAWS   = 1000
N_TUNE    = 500
N_CHAINS  = 2
N_CORES   = 1
TARGET_ACCEPT = 0.9
RANDOM_SEED   = 42


with M0:
    trace_M0 = pm.sample(
    draws         = N_DRAWS,
    tune          = N_TUNE,
    chains        = N_CHAINS,
    cores         = N_CORES,
    target_accept = TARGET_ACCEPT,
    random_seed   = RANDOM_SEED,
    idata_kwargs  = {'log_likelihood': True},
    )

# print(az.summary(trace_M0, var_names=['pi', 'mu_slab', 'sigma_slab', 'nu', 'sigma_obs']))

# %%

# Global parameters only — z has 100*10 = 1000 dimensions, don't plot those
az.plot_trace(trace_M0, var_names=['mu_slab', 'sigma_slab', 'sigma_obs'])
plt.suptitle('Traces — M0 simple Normal prior test')
plt.tight_layout()
plt.show()

az.summary(trace_M0, var_names=['mu_slab', 'sigma_slab', 'sigma_obs'])

# %%

# with M0:
#     approx   = pm.fit(n=10000, method='advi',
#                       progressbar=True)
#     trace_M0 = approx.sample(1000)

    # %%
# ── Sample posterior predictive ───────────────────────────────────────────────
with M0:
    post_pred = pm.sample_posterior_predictive(trace_M0)

# ── Extract ───────────────────────────────────────────────────────────────────
z_post    = trace_M0.posterior['z'].values        # (chains, draws, N_AREAS, n_years)
P_post    = post_pred.posterior_predictive['P_like'].values
E_post    = post_pred.posterior_predictive['E_like'].values

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ── 1. Posterior predictive vs observed: planning ─────────────────────────────
ax = axes[0, 0]
ax.hist(P_post.reshape(-1), bins=200, density=True,
        alpha=0.5, color='steelblue', label='Posterior predictive')
ax.hist(P_sub.ravel(), bins=200, density=True,
        alpha=0.5, color='coral', label='Observed')
ax.set_xlim(-150, 150)
ax.set_xlabel('Net dwelling change')
ax.set_title('Posterior predictive vs observed: planning')
ax.legend()

# ── 2. Posterior predictive vs observed: BEN ──────────────────────────────────
ax = axes[0, 1]
ax.hist(E_post.reshape(-1), bins=200, density=True,
        alpha=0.5, color='steelblue', label='Posterior predictive')
ax.hist(E_sub.ravel(), bins=200, density=True,
        alpha=0.5, color='coral', label='Observed')
ax.set_xlim(-150, 150)
ax.set_xlabel('Net dwelling change')
ax.set_title('Posterior predictive vs observed: BEN')
ax.legend()

# ── 3. Posterior mean z vs observed means by year ─────────────────────────────
ax = axes[0, 2]
z_year_mean  = z_post.mean(axis=(0, 1, 2))
z_year_lower = np.percentile(z_post.mean(axis=2), 5,  axis=(0, 1))
z_year_upper = np.percentile(z_post.mean(axis=2), 95, axis=(0, 1))

ax.plot(infer_years, z_year_mean, marker='o', color='black',
        label='Posterior mean z')
ax.fill_between(infer_years, z_year_lower, z_year_upper,
                alpha=0.2, color='black', label='90% CI on mean')
ax.plot(infer_years, P_sub.mean(axis=0), marker='s',
        color='steelblue', label='Planning mean')
ax.plot(infer_years, E_sub.mean(axis=0), marker='^',
        color='coral', label='BEN mean')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean net dwelling change')
ax.set_title('Posterior mean z vs observed means by year')
ax.legend(fontsize=8)

# ── 4. Census constraint check ────────────────────────────────────────────────
ax = axes[1, 0]
z_sums    = z_post.sum(axis=-1).reshape(-1, N_AREAS)
residuals = (z_sums - D_sub[None, :]).ravel()
ax.hist(residuals, bins=100, density=True, color='steelblue', alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('z sum - D')
ax.set_title('Posterior census constraint violations')
ax.text(0.05, 0.95,
        f'mean={np.abs(residuals).mean():.3f}\nmax={np.abs(residuals).max():.3f}',
        transform=ax.transAxes, verticalalignment='top', fontsize=9)

# ── 5. Posterior z for sample areas ───────────────────────────────────────────
ax = axes[1, 1]
sample_idx = np.random.choice(N_AREAS, 6, replace=False)
for idx in sample_idx:
    z_mean = z_post[:, :, idx, :].mean(axis=(0, 1))
    ax.plot(infer_years, z_mean, alpha=0.6, marker='o', markersize=3)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Posterior mean z')
ax.set_title('Posterior mean z for 6 sample areas')

# ── 6. Posterior on global parameters ─────────────────────────────────────────
ax = axes[1, 2]
for param, color, label in zip(
    ['mu_slab', 'sigma_slab', 'sigma_obs'],
    ['steelblue', 'coral', 'green'],
    ['mu_slab', 'sigma_slab', 'sigma_obs']
):
    samples = trace_M0.posterior[param].values.ravel()
    ax.hist(samples, bins=50, density=True, alpha=0.5,
            color=color, label=label)
ax.set_xlabel('Value')
ax.set_title('Posterior: global parameters')
ax.legend()

plt.suptitle('Posterior predictive checks — M0 simple Normal prior')
plt.tight_layout()
plt.show()

# %%

# ── Select areas spanning the range of census differences ─────────────────────
n_sample    = 6
D_sub_series = pd.Series(D_sub)
quantiles   = np.linspace(0, 1, n_sample)
sample_idx  = [
    D_sub_series.sub(D_sub_series.quantile(q)).abs().idxmin()
    for q in quantiles
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, idx in zip(axes.ravel(), sample_idx):
    z_area      = z_post[:, :, idx, :]
    z_mean      = z_area.mean(axis=(0, 1))
    z_lo        = np.percentile(z_area, 5,  axis=(0, 1))
    z_hi        = np.percentile(z_area, 95, axis=(0, 1))
    z_sum_mean  = z_mean.sum()
    z_sum_lo    = z_area.sum(axis=-1).reshape(-1).mean() - 2 * z_area.sum(axis=-1).reshape(-1).std()
    z_sum_hi    = z_area.sum(axis=-1).reshape(-1).mean() + 2 * z_area.sum(axis=-1).reshape(-1).std()

    ax.set_title(f'LSOA {idx}  (census diff={D_sub[idx]:.0f}  '
             f'post. sum={z_sum_mean:.0f} '
             f'[{z_sum_lo:.0f}, {z_sum_hi:.0f}])')
    
    ax.plot(infer_years, z_mean, color='black', marker='o',
            linewidth=1.5, label='Posterior mean z')
    ax.fill_between(infer_years, z_lo, z_hi,
                    alpha=0.2, color='black', label='90% CI')
    ax.plot(infer_years, P_sub[idx], color='steelblue', marker='s',
            alpha=0.7, linewidth=1.0, label='Planning')
    ax.plot(infer_years, E_sub[idx], color='coral', marker='^',
            alpha=0.7, linewidth=1.0, label='BEN')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

    ax.set_xlabel('Year')
    ax.set_ylabel('Net dwelling change')

    # ax.spines[['top', 'right']].set_visible(False)

    if idx == sample_idx[0]:
        ax.legend(fontsize=10)

plt.suptitle('Posterior mean z vs planning and current estimates — spanning census diff range')
plt.tight_layout()
plt.show()

# %%

z_mean_post = z_post.mean(axis=(0, 1))   # (N_AREAS, n_years)

resid_plan = P_sub - z_mean_post
resid_ben  = E_sub - z_mean_post

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, resid, label in zip(axes, [resid_plan, resid_ben], ['Planning', 'BEN']):
    ax.hist(resid.ravel(), bins=100, density=True, color='steelblue', alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(resid.mean(), color='red', linestyle='--',
               linewidth=0.8, label=f'mean={resid.mean():.2f}')
    ax.set_title(f'Residuals: {label}')
    ax.set_xlabel('Observed - posterior mean z')
    ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_years, resid_plan.mean(axis=0), marker='o',
        color='steelblue', label='Planning')
ax.plot(infer_years, resid_ben.mean(axis=0),  marker='^',
        color='coral',     label='BEN')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean residual')
ax.set_title('Mean residuals by year')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, resid, label in zip(axes, [resid_plan, resid_ben], ['Planning', 'BEN']):
    ax.scatter(D_sub, resid.mean(axis=1), alpha=0.3, s=5, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Census diff (D)')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Residuals vs census diff: {label}')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, resid, label in zip(axes, [resid_plan, resid_ben], ['Planning', 'BEN']):
    ax.scatter(D_sub, resid.mean(axis=1), alpha=0.3, s=5, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Census diff (D)')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Residuals vs census diff: {label}')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, obs, label in zip(axes, [P_sub, E_sub], ['Planning', 'BEN']):
    ax.scatter(z_mean_post.ravel(), obs.ravel(),
               alpha=0.1, s=3, color='steelblue')
    lims = [min(z_mean_post.min(), obs.min()),
            max(z_mean_post.max(), obs.max())]
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Posterior mean z')
    ax.set_ylabel(f'{label} observed')
    ax.set_title(f'Posterior z vs {label}')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

# Only look at LSOA-years where at least one source is meaningfully non-zero
threshold = 5   # at least one source shows > 5 dwellings change
mask = (np.abs(P_sub) > threshold) | (np.abs(E_sub) > threshold)

source_disagreement = np.abs(P_sub - E_sub)
posterior_ci_width  = (np.percentile(z_post, 95, axis=(0, 1)) -
                       np.percentile(z_post, 5,  axis=(0, 1)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(source_disagreement[mask], posterior_ci_width[mask],
           alpha=0.2, s=5, color='steelblue')
ax.set_xlabel('|Planning - BEN| (source disagreement)')
ax.set_ylabel('Posterior 90% CI width')
ax.set_title(f'Posterior uncertainty vs source disagreement (|obs|>{threshold})')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


from esda.moran import Moran
from libpysal.weights import Queen

mean_resid_plan = resid_plan.mean(axis=1)   # (N_AREAS,)
mean_resid_ben  = resid_ben.mean(axis=1)

w = Queen.from_dataframe(dwellings_all.iloc[:N_AREAS], use_index=False)
w.transform = 'r'

moran_plan = Moran(mean_resid_plan, w)
moran_ben  = Moran(mean_resid_ben,  w)

print(f"Moran's I (planning residuals): {moran_plan.I:.4f}  p={moran_plan.p_sim:.4f}")
print(f"Moran's I (BEN residuals):      {moran_ben.I:.4f}  p={moran_ben.p_sim:.4f}")


z_lo = np.percentile(z_post, 5,  axis=(0, 1))
z_hi = np.percentile(z_post, 95, axis=(0, 1))

for obs, label in zip([P_sub, E_sub], ['Planning', 'BEN']):
    coverage = np.mean((obs >= z_lo) & (obs <= z_hi))
    print(f'{label} coverage of 90% CI: {coverage:.3f}')

# %%

z_lo = np.percentile(z_post, 5,  axis=(0, 1))
z_hi = np.percentile(z_post, 95, axis=(0, 1))

for obs, label in zip([P_sub, E_sub], ['Planning', 'BEN']):
    coverage = np.mean((obs >= z_lo) & (obs <= z_hi))
    print(f'{label} coverage of 90% CI: {coverage:.3f}')

# %%
