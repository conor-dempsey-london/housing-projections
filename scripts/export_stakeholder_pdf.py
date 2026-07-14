"""
Renders docs/az3-stakeholder-summary.md to a PDF for circulating outside the
repo. No new heavy dependency: converts Markdown -> HTML with the `markdown`
package (already a transitive dependency), then shells out to headless Edge
(ships with Windows) for the actual HTML -> PDF render via
`--print-to-pdf`, rather than adding weasyprint/wkhtmltopdf/pandoc.

Re-run this after any edit to the source Markdown; it always overwrites the
PDF from scratch.

Usage
-----
    pixi run python scripts/export_stakeholder_pdf.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

_REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD = _REPO_ROOT / 'docs' / 'az3-stakeholder-summary.md'
OUTPUT_PDF = _REPO_ROOT / 'docs' / 'az3-stakeholder-summary.pdf'
_SCRATCH_HTML = _REPO_ROOT / 'results' / 'scratch' / 'az3-stakeholder-summary_render.html'

EDGE_CANDIDATES = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
]

PAGE_CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 720px;
       margin: 2em auto; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
h2 { font-size: 1.25em; margin-top: 1.6em; border-bottom: 1px solid #ccc; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #999; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f0f0f0; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
strong { color: #000; }
"""


def _find_edge():
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which('msedge')
    if found:
        return found
    raise RuntimeError('Could not find msedge.exe -- adjust EDGE_CANDIDATES '
                        'or install a headless-capable browser.')


def main():
    md_text = SOURCE_MD.read_text(encoding='utf-8')
    body_html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    full_html = f'<!doctype html><html><head><meta charset="utf-8">' \
                f'<style>{PAGE_CSS}</style></head><body>{body_html}</body></html>'
    _SCRATCH_HTML.parent.mkdir(parents=True, exist_ok=True)
    _SCRATCH_HTML.write_text(full_html, encoding='utf-8')

    edge = _find_edge()
    cmd = [
        edge, '--headless', '--disable-gpu', '--no-sandbox',
        f'--print-to-pdf={OUTPUT_PDF}',
        '--print-to-pdf-no-header',
        _SCRATCH_HTML.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not OUTPUT_PDF.exists():
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError('PDF export failed -- see stdout/stderr above.')

    print(f'Wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1e3:.0f} KB)')


if __name__ == '__main__':
    main()
