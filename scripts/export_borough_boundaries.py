"""
One-time export of borough-level boundary geometry for the AZ3 estimates
dashboard (docs/estimates-dashboard-report-plan.md Phase B). Runs against
this repo's normal (private-data-access) environment -- the dashboard
notebook itself never imports geopandas/gla_data/housing_projections at
runtime, since it's exported to a self-contained static HTML-WASM file for
non-technical stakeholders (no server, no private dataset access in-browser).

Dissolves LSOAs to borough polygons (avoids shipping 4987 detailed LSOA
boundaries in a client-side bundle), simplifies, reprojects OSGB36/BNG
(EPSG:27700) to WGS84 (EPSG:4326) for a plain GeoJSON any web map / Altair
mark_geoshape can read with no further geo-processing.

Usage
-----
    pixi run python scripts/export_borough_boundaries.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import gla_data._ons  # noqa: E402
from housing_projections.config import DATA_PATH  # noqa: E402
from housing_projections.data import load_data, make_borough_idx  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402

OUTPUT_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'artifacts'
               / 'az3_year_estimates' / 'borough_boundaries.geojson')
SIMPLIFY_TOLERANCE_M = 20  # metres, in the original BNG projection


def main():
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

    boroughs = gdf.dissolve(by='borough_name', as_index=False)[['borough_name', 'geometry']]
    boroughs['geometry'] = boroughs['geometry'].simplify(SIMPLIFY_TOLERANCE_M,
                                                            preserve_topology=True)
    boroughs = boroughs.to_crs(4326)

    geojson = json.loads(boroughs.to_json())
    OUTPUT_PATH.write_text(json.dumps(geojson))
    print(f'Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e3:.0f} KB, '
          f'{len(boroughs)} boroughs)')


if __name__ == '__main__':
    main()
