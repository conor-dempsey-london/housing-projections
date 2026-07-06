# housing-projections

Bayesian dwelling projection models for London LSOAs.

## Setup

Requires [pixi](https://pixi.sh). Clone and install:

```bash
git clone https://github.com/conor-dempsey-london/housing-projections
cd housing-projections
pixi install
```

Copy `.env.example` to `.env` and fill in your paths.

## Usage

```bash
pixi run run-models    # sample all models
pixi run compare       # LOO comparison and diagnostics
pixi run report        # generate HTML report
pixi run test          # run test suite
```
