# dwelling-estimates

Bayesian estimation of dwelling delivery rates for London LSOAs from noisy administrative data sources.

Requires access to [pld-database-live](https://github.com/JamesB686/pld_database_live).

```bash
git clone https://github.com/conor-dempsey-london/housing-projections
cd housing-projections
pixi install
cp .env.example .env  # fill in paths
```

```bash
pixi run run-models
pixi run compare
pixi run report
```
