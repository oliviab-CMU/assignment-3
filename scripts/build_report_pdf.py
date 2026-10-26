"""Render REPORT.md to report.pdf.

Markdown -> HTML (markdown_it) -> PDF (headless Edge/Chrome --print-to-pdf).

The report is written in Markdown so it stays diffable; the assignment wants a
PDF. This script is the only build step, so re-running it after an edit is the
whole publishing workflow:

    uv run python scripts/build_report_pdf.py

A handful of inline LaTeX math spans appear in the prose. They are mapped to
HTML explicitly rather than pulled through a math renderer: the report has
four of them, and a general TeX pipeline would add a large dependency (and
MathJax's network fetch) for no reader benefit at this size.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "REPORT.md"
OUTPUT = REPO_ROOT / "report.pdf"

# Inline math, in the exact forms that appear in REPORT.md.
MATH = {
    r"$A = r$": "<em>A</em> = <em>r</em>",
    r"$A_i = r_i - \frac{1}{G}\sum_k r_k$": (
        "<em>A</em><sub><em>i</em></sub> = "
        "<em>r</em><sub><em>i</em></sub> − (1/<em>G</em>) "
        "Σ<sub><em>k</em></sub> <em>r</em><sub><em>k</em></sub>"
    ),
    r"$G$": "<em>G</em>",
    r"$\rho = \pi_\text{new}(a \mid s) / \pi_\text{old}(a \mid s)$": (
        "ρ = π<sub>new</sub>(<em>a</em> ∣ <em>s</em>) / "
        "π<sub>old</sub>(<em>a</em> ∣ <em>s</em>)"
    ),
}

BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.55; color: #17181a;
  margin: 0; max-width: none;
}
h1 { font-size: 21pt; line-height: 1.25; margin: 0 0 6pt; letter-spacing: -0.2pt; }
h2 { font-size: 14.5pt; margin: 22pt 0 7pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid #d9d7d2; break-after: avoid; }
h3 { font-size: 11.6pt; margin: 15pt 0 5pt; break-after: avoid; }
h4 { font-size: 10.4pt; margin: 12pt 0 4pt; break-after: avoid; font-style: italic; }
p, li { orphans: 2; widows: 2; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
strong { color: #000; }
code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", monospace;
  font-size: 0.88em; background: #f2f1ee; padding: 0.5pt 2.5pt;
  border-radius: 2pt;
}
pre {
  background: #f5f4f1; border: 0.6pt solid #e2e0db; border-radius: 3pt;
  padding: 7pt 9pt; font-size: 8.4pt; line-height: 1.42;
  overflow-wrap: anywhere; white-space: pre-wrap; break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; font-size: 9pt;
  margin: 4pt 0 10pt; break-inside: avoid;
}
th { text-align: left; font-weight: 600; padding: 4pt 6pt;
     border-bottom: 1pt solid #b9b6b0; }
td { padding: 3.5pt 6pt; border-bottom: 0.5pt solid #e6e4df;
     vertical-align: top; }
tbody tr:last-child td { border-bottom: 1pt solid #b9b6b0; }
img {
  max-width: 100%; height: auto; display: block; margin: 8pt auto 4pt;
  break-inside: avoid;
}
hr { border: none; border-top: 0.7pt solid #ddd; margin: 16pt 0; }
blockquote {
  margin: 0 0 9pt; padding: 5pt 12pt; border-left: 2.5pt solid #c9c5bd;
  color: #45464a; background: #faf9f7;
}
em { font-style: italic; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>
{body}
</body></html>
"""


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit(
        "No Chromium-family browser found for PDF printing. Install Chrome or "
        "Edge, or print REPORT.md to PDF another way."
    )


def render_html(markdown_text: str) -> str:
    for tex, html in MATH.items():
        markdown_text = markdown_text.replace(tex, html)
    leftovers = re.findall(r"\$[^$\n]{1,80}\$", markdown_text)
    if leftovers:
        raise SystemExit(f"Unmapped inline math in REPORT.md: {leftovers}")

    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    body = md.render(markdown_text)
    title = re.search(r"^#\s+(.+)$", markdown_text, re.M)
    return TEMPLATE.format(
        title=title.group(1) if title else "Report",
        css=CSS,
        body=body,
    )


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")

    html = render_html(SOURCE.read_text(encoding="utf-8"))
    browser = find_browser()

    # The HTML must sit next to the markdown so relative figure paths resolve.
    with tempfile.TemporaryDirectory(
        dir=REPO_ROOT, prefix="_report_build_"
    ) as tmp:
        tmp_dir = Path(tmp)
        html_path = tmp_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")
        # Figure paths in the report are relative (`report_plots/x.png`), so the
        # build directory needs the same subdirectory, not a flat copy.
        shutil.copytree(REPO_ROOT / "report_plots", tmp_dir / "report_plots")
        pdf_path = tmp_dir / "report.pdf"

        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=8000",
                f"--print-to-pdf={pdf_path}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not pdf_path.exists():
            raise SystemExit("browser did not produce a PDF")
        shutil.move(str(pdf_path), OUTPUT)

    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
          f"({OUTPUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    sys.exit(main())
