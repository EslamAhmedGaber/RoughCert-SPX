"""Unified Q1 publication figure style for RoughCert-SPX (Paper 2).

Exact print width: elsarticle [final,3p,11pt] has \\textwidth = 468.00 pt = 6.476 in (measured with
measure_textwidth.py; the two-column edition sets figure* at 522 pt, a 1.115 enlargement).
Every figure is drawn at exact print width (zero LaTeX rescaling) with one
consistent colour-blind-safe editorial palette, Computer Modern mathtext, clean
despined axes, and fixed method identity across all figures.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt

# Exact print geometry from pdflatex (\the\textwidth = 468.00pt in the one-column edition)
TEXTWIDTH_IN = 468.0 / 72.27

# Fixed method colour identity across the ENTIRE paper (consistent with Paper 1)
PALETTE = {
    "roughcert": "#0F766E",       # Deep editorial teal (hero RoughCert-SPX-CE / Bridge)
    "roughcert_fill": "#14B8A6",  # Translucent teal band fill
    "roughcert_light": "#CCFBF1", # Light teal tint
    "det": "#047857",             # Emerald pine (176-bit Arb RoughCert-SPX-Det)
    "det_fill": "#A7F3D0",        # Soft emerald tint
    "jensen": "#BE123C",          # Deep rose (ce_adapted Jensen-J ablation / pitfall)
    "bht": "#1D4ED8",             # Cobalt navy (BHT2020-BridgeRQMC / Left-Point)
    "mp2018": "#7C3AED",          # Royal violet (MP2018-Hybrid)
    "markov": "#D97706",          # Burnished amber (BayerBreneis2023 Markovian)
    "rdonsker": "#DC2626",        # Crimson vermilion (HJMS2024-rDonsker)
    "rdonsker_b": "#EA580C",      # Warm rust orange (HJMS2024-rDonsker b_k* weights)
    "bg2012": "#475569",          # Cool slate (BergomiGuyon2012-O2)
    "exact": "#0F172A",           # Near-black ink (exact analytical law)
    "mixed": "#0284C7",           # Ocean blue (Mixed rBergomi)
    "tol": "#BE123C",             # Deep rose threshold line (10/25 bps)
    "tol_fill": "#FFE4E6",        # Soft rose shading
    "grid": "#E2E8F0",            # Faint neutral hairline grid
    "ink": "#1E293B",             # Dark charcoal spine / primary ink
    "muted": "#64748B",           # Secondary annotation / reference ink
    "gold_fill": "#FEF3C7",       # Soft amber highlight
}

H_COLORS = {
    "0.05": "#DC2626",
    "0.07": "#EA580C",
    "0.10": "#1D4ED8",
    "0.15": "#7C3AED",
    "0.25": "#0F766E",
    "0.30": "#0F766E",
    "0.40": "#D97706",
}


def apply_style(base_size: float = 8.5) -> None:
    """Apply clean editorial Q1 Matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": base_size,
        "axes.titlesize": base_size + 0.3,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": base_size,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base_size - 0.8,
        "ytick.labelsize": base_size - 0.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "legend.fontsize": base_size - 0.9,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.55,
        "grid.linestyle": "-",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": PALETTE["ink"],
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    """Apply subtle background grid and crisp despined aesthetics."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major", color=PALETTE["grid"], lw=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])


def panel_title(ax, letter: str, subtitle: str) -> None:
    """Bold lowercase panel label followed by crisp plain subtitle."""
    ax.set_title(f"$\\mathbf{{({letter})}}$  {subtitle}", loc="left", pad=6.5)


def tex_sci(v: float) -> str:
    """Format scientific notation cleanly as LaTeX math (no 1e-05 strings)."""
    if v == 0:
        return "$0$"
    m, e = f"{v:.1e}".split("e")
    e_int = int(e)
    m_f = float(m)
    if abs(m_f - 1.0) < 1e-9:
        return f"$10^{{{e_int}}}$"
    if abs(m_f - round(m_f)) < 1e-9:
        return f"${int(round(m_f))}\\times 10^{{{e_int}}}$"
    return f"${m_f:.1f}\\times 10^{{{e_int}}}$"


def save_figure(fig, out_dir: Path, stem: str, also_png: bool = True) -> list[Path]:
    """Save vector PDF at exact TEXTWIDTH_IN and 240 DPI PNG preview for visual QA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.045)
    with fitz.open(stream=buf.getvalue(), filetype="pdf") as src, fitz.open() as out:
        box = src[0].rect
        page = out.new_page(width=72.0 * TEXTWIDTH_IN, height=72.0 * TEXTWIDTH_IN * box.height / box.width)
        page.show_pdf_page(page.rect, src, 0)
        out.save(pdf_path, garbage=4, deflate=True)
        paths = [pdf_path]
        if also_png:
            png_dir = out_dir.parents[1] / "audit" / "figure_previews"
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"{stem}.png"
            page.get_pixmap(matrix=fitz.Matrix(240 / 72, 240 / 72)).save(png_path)
            paths.append(png_path)
    plt.close(fig)
    return paths
