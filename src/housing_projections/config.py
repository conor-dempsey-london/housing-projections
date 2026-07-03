
# ── Inference years ───────────────────────────────────────────────────────────
INFER_YEARS      = list(range(2012, 2022))
N_YEARS          = len(INFER_YEARS)

# ── Column naming conventions ─────────────────────────────────────────────────
INFER_COLS_PLAN  = [f'{y}/{str(y+1)[-2:]}' for y in range(2011, 2021)]
INFER_COLS_BEN   = [f'{y}_ben'              for y in range(2011, 2021)]

ALL_COLS_PLAN    = [f'{y}/{str(y+1)[-2:]}' for y in range(2009, 2025)]
ALL_COLS_BEN     = [f'{y}_ben'              for y in range(2009, 2025)]

CENSUS_COLS      = ['dwellings_2011', 'dwellings_2021']

# ── Census constraint ─────────────────────────────────────────────────────────
CENSUS_REL_ERROR = 0.02
CENSUS_ABS_FLOOR = 2.0

# ── Sampling defaults ─────────────────────────────────────────────────────────
DEFAULT_SAMPLE_KWARGS = dict(
    draws         = 1500,
    tune          = 500,
    chains        = 2,
    cores         = 1,
    target_accept = 0.9,
    random_seed   = 42,
)

# ── Plot colours ──────────────────────────────────────────────────────────────
COLOURS = {
    'z':         'black',
    'planning':  'steelblue',
    'ben':       'coral',
    'baseline':  'green',
    'posterior': 'purple',
}
