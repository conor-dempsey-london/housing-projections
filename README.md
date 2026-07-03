# housing-projections

Tools for housing projection modelling at the GLA.

## Installation

**This package is not pure Python.** It depends on [geopandas](https://geopandas.org), which in turn depends on native C/C++ libraries (GDAL, PROJ) that cannot be reliably installed via pip alone. For this reason, **[pixi](https://pixi.sh) is the required package manager** for any project that uses this package.

Pixi resolves the geospatial stack from [conda-forge](https://conda-forge.org), which maintains pre-built binaries for all platforms and Python versions. This sidesteps the wheel availability problems that affect PyPI for these libraries.

### Setting up a new project that depends on this package

1. [Install pixi](https://pixi.sh/latest/#installation) if you haven't already.

2. Initialise your project:

    ```bash
    pixi init my-project
    cd my-project
    ```

3. Add the conda-forge geospatial dependencies:

    ```bash
    pixi add fiona geopandas
    ```

4. Add this package as a PyPI git dependency. In your `pyproject.toml` (or `pixi.toml`), add:

    ```toml
    [tool.pixi.pypi-dependencies]
    housing-projections = { git = "https://github.com/Greater-London-Authority/housing-projections", branch = "master" }
    ```

    Then run:

    ```bash
    pixi install
    ```

### Why not pip?

The geospatial stack (geopandas → fiona/pyogrio → GDAL, pyproj → PROJ) relies on large native libraries that each Python package must rebuild and vendor when publishing to PyPI. This process is fragile, platform-specific, and often lags new Python releases by months. conda-forge solves this by maintaining a shared, coordinated build of these libraries across the whole ecosystem.

Attempting to install this package with pip directly will likely fail with GDAL- or PROJ-related errors unless you have already installed those system libraries separately.

## Development

Clone the repository and install in editable mode with pixi:

```bash
git clone https://github.com/Greater-London-Authority/housing-projections
cd housing-projections
pixi install
```

Run the test suite:

```bash
pixi run test
```
