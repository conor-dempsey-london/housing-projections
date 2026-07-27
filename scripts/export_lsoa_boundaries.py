"""
One-time export of LSOA-level boundary geometry for the AZ3 estimates
dashboard's choropleth drill-down (borough -> LSOA-grain map). Sibling of
export_borough_boundaries.py -- same environment/runtime split rationale (this
script needs geopandas/gla_data/housing_projections; the dashboard notebook
itself never does, since it's exported to a self-contained static HTML-WASM
file for non-technical stakeholders).

Unlike the borough export, this keeps LSOA grain rather than dissolving --
that's the whole point of the drill-down -- but simplifies at a finer
tolerance than the borough export's 20m, since LSOAs are far smaller and 20m
visibly over-simplifies small urban ones. Reprojects BNG (EPSG:27700) to WGS84
(EPSG:4326), same as the borough export, for plain Altair mark_geoshape
consumption.

Usage
-----
    pixi run python scripts/export_lsoa_boundaries.py [--tolerance-m 10]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import gla_data._ons  # noqa: E402
from housing_projections.config import DATA_PATH  # noqa: E402
from housing_projections.data import load_data, make_borough_idx  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / 'results' / 'artifacts'
              / 'az3_year_estimates')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tolerance-m', type=float, default=10,
                     help='simplification tolerance in metres, BNG projection')
    ap.add_argument('--output', default=str(OUTPUT_DIR / 'lsoa_boundaries.geojson'))
    args = ap.parse_args()

    gdf = load_data(str(DATA_PATH))
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)

    borough_idx, n_boroughs, borough_codes = make_borough_idx(gdf)
    geo_lookup = gla_data._ons.fetch_geography_lookup(2021, 'lsoa')
    geo_lookup = geo_lookup[geo_lookup['LSOA21CD'].isin(gdf['LSOA21CD'])].copy()
    code_to_name = (geo_lookup.drop_duplicates('LAD22CD')
                     .set_index('LAD22CD')['LAD22NM'].to_dict())

    gdf = gdf.copy()
    gdf['borough_code'] = [borough_codes[i] for i in borough_idx]
    gdf['borough_name'] = gdf['borough_code'].map(code_to_name)

    lsoas = gdf[['LSOA21CD', 'borough_name', 'geometry']].copy()
    lsoas['geometry'] = lsoas['geometry'].simplify(args.tolerance_m, preserve_topology=True)
    lsoas = lsoas.to_crs(4326)

    geojson = json.loads(lsoas.to_json())
    output_path = Path(args.output)
    output_path.write_text(json.dumps(geojson))
    print(f'Wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB, '
          f'{len(lsoas)} LSOAs, tolerance={args.tolerance_m}m)')


if __name__ == '__main__':
    main()
