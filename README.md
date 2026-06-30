## Usage

We recommend managing dependencies with [pixi](https://pixi.sh), which handles
the GDAL system dependency required by this package automatically via conda-forge.

First install Fiona, which will pull in GDAL:

```bash
pixi add fiona
```

Then add this package:

```bash
pixi add housing-projections
```

If you install without adding Fiona first, you may encounter GDAL-related errors
during installation.