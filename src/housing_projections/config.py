
import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = three levels up from src/housing_projections/config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_REPO_ROOT / '.env')

# ── Project paths ─────────────────────────────────────────────────────────────
# Set DATA_PATH in a .env file at the repo root (see .env.example).
# The remaining paths default to subdirectories of the repo root so they work
# regardless of which directory the process (or notebook kernel) is started from.

_data_path  = os.getenv('DATA_PATH')
DATA_PATH   = Path(_data_path) if _data_path else None   # None until set in .env
RESULTS_DIR = Path(os.getenv('RESULTS_DIR',  str(_REPO_ROOT / 'results')))
TRACES_DIR  = Path(os.getenv('TRACES_DIR',   str(_REPO_ROOT / 'results' / 'traces')))
REPORT_PATH = Path(os.getenv('REPORT_PATH',  str(_REPO_ROOT / 'results' / 'report.html')))

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
    chains        = 4,
    cores         = 4,
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
