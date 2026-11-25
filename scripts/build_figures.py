"""Build all 7 publication-grade vector PDF figures (and 240 DPI PNG previews); every literal font size is at least 7.4 pt at print width for RoughCert-SPX.

Every figure is rendered at exact print width TEXTWIDTH_IN = 468 pt = 6.476 in using spx_style.py
directly from verified results/e1_exact_laws.json .. results/e6_joint_spx_vix.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fractions import Fraction
from scripts.build_numbers_and_tables import dround
from scripts.spx_style import (
    PALETTE,
    H_COLORS,
    TEXTWIDTH_IN,
    apply_style,
    style_axes,
    panel_title,
    save_figure,
    tex_sci,
)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RES = ROOT / "results"
FIG_DIR = ROOT / "manuscript" / "figures"


def plain_log_ticks(ax, xt=None, yt=None, yfmt=None):
    """Fixed, readable ticks on log axes (no overlapping minor-tick labels)."""
    if xt is not None:
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
    if yt is not None:
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt if yfmt is not None else (lambda v, _: f"{v:g}")))
        ax.yaxis.set_minor_locator(NullLocator())


def load_json(name: str) -> dict:
    p = RES / name
    if not p.exists():
        sys.exit(f"FATAL: Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Figure 1: E1 Exact-Law Validation (Correlated Rough Stein-Stein)
# -----------------------------------------------------------------------------
def build_fig_e1():
    e1 = load_json("e1_exact_laws.json")
    strikes = np.array(e1["strikes"])
    c1 = e1["containment_summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.80), gridspec_kw={"wspace": 0.34})

    # Panel (a): Signed IV bias vs strike at n=32 (dashed) and n=256 (solid)
    specs = [
        ("ce_i_only", r"$\mathsf{RoughCert\text{-}SPX\text{-}CE}$", PALETTE["roughcert"], "o", 1.7),
        ("leftpoint", "Left-point (FSW)", PALETTE["bht"], "s", 1.3),
        ("ce_adapted", r"Jensen-$J$ ablation", PALETTE["jensen"], "^", 1.3),
    ]
    ax1.axhline(0.0, color=PALETTE["ink"], lw=0.7, ls=":", zorder=1)
    for key, label, col, mk, lw in specs:
        b32 = c1[key]["iv_bias_by_level"]["32"]["errs_bps"]
        b256 = c1[key]["iv_bias_by_level"]["256"]["errs_bps"]
        ax1.plot(strikes, b32, color=col, ls="--", lw=lw * 0.85, marker=mk, ms=3.5, alpha=0.65)
        ax1.plot(strikes, b256, color=col, ls="-", lw=lw, marker=mk, ms=4.2, label=label)
    ax1.plot([], [], color=PALETTE["muted"], ls="--", lw=1.0, label=r"$n=32$ (dashed)")
    ax1.plot([], [], color=PALETTE["muted"], ls="-", lw=1.2, label=r"$n=256$ (solid)")
    ax1.set_xlabel(r"Strike $K$")
    ax1.set_ylabel("IV bias vs exact Fourier (bps)")
    ax1.set_xticks(strikes)
    ax1.set_xticklabels([f"{k:.2f}" for k in strikes], fontsize=7.4)
    style_axes(ax1, "y")
    panel_title(ax1, "a", "Signed smile IV bias (rough Stein\u2013Stein)")

    # Panel (b): Max |IV bias| vs grid n (log-log) with FSW & 2H reference lines
    levels = np.array([32, 64, 128, 256], dtype=float)
    for key, label, col, mk, lw in specs:
        vals = [c1[key]["iv_bias_by_level"][str(int(n))]["max_abs_bps"] for n in levels]
        ax2.loglog(levels, vals, color=col, lw=lw, marker=mk, ms=4.5, label=label)

    # Reference slopes placed cleanly above data (labelled with axis variable n)
    n_ref = np.array([32.0, 256.0])
    y_fsw = 38.0 * (n_ref / 32.0) ** (-0.8)
    y_2h = 46.0 * (n_ref / 32.0) ** (-0.2)
    ref_fsw, = ax2.loglog(n_ref, y_fsw, color=PALETTE["muted"], ls=":", lw=1.0, label=r"$n^{-0.8}$ ($\beta_{\mathrm{FSW}}$)")
    ref_2h, = ax2.loglog(n_ref, y_2h, color=PALETTE["muted"], ls="--", lw=0.9, alpha=0.85, label=r"$n^{-0.2}$ ($2H$)")
    ax2.legend(handles=[ref_fsw, ref_2h], loc="lower left", fontsize=7.4, handlelength=2.2, borderaxespad=0.3)

    ax2.set_xlabel(r"Time steps $n$")
    ax2.set_ylabel(r"Max $|\mathrm{IV\ bias}|$ across smile (bps)")
    ax2.set_xticks(levels)
    ax2.set_xticklabels([str(int(n)) for n in levels])
    ax2.minorticks_off()
    ax2.set_ylim(4.0, 65.0)
    plain_log_ticks(ax2, yt=[5, 10, 20, 40])
    style_axes(ax2, "both")
    panel_title(ax2, "b", "Convergence vs exact law ($H=0.10$)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, fontsize=7.4)
    return save_figure(fig, FIG_DIR, "fig_e1_exact_rss")


# -----------------------------------------------------------------------------
# Figure 2: E2 Deterministic Certifiability Phase Map
# -----------------------------------------------------------------------------
def build_fig_e2():
    e2 = load_json("e2_det_phase_map.json")
    cells_10 = [c for c in e2["phase_cells"] if abs(c["H"] - 0.10) < 1e-9]
    eta_eff = np.array([c["eta_eff"] for c in cells_10])
    w_r0_2m = np.array([c["iv_width_bps_rho0_2m"] for c in cells_10])
    w_r0_k0 = np.array([c["iv_width_bps_rho0_k0"] for c in cells_10])
    w_r5 = np.array([c["iv_width_bps_rho_m05"] for c in cells_10])
    w_r9 = np.array([c["iv_width_bps_rho_m09"] for c in cells_10])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.85), gridspec_kw={"wspace": 0.34})

    # Panel (a): Arb IV width vs eta*T^H at H=0.10
    ax1.loglog(eta_eff, w_r0_2m, color=PALETTE["det"], lw=1.8, marker="o", ms=4.2, label=r"$\rho=0$ (2-moment sharp)")
    ax1.loglog(eta_eff, w_r0_k0, color=PALETTE["bg2012"], lw=1.1, ls="--", marker="d", ms=3.4, label=r"$\rho=0$ ($k=0$ Lipschitz)")
    ax1.loglog(eta_eff, w_r5, color=PALETTE["markov"], lw=1.35, marker="s", ms=3.8, label=r"$\rho=-0.5$ ($k=0$)")
    ax1.loglog(eta_eff, w_r9, color=PALETTE["jensen"], lw=1.5, marker="^", ms=4.0, label=r"$\rho=-0.9$ ($k=0$)")
    ax1.axhline(25.0, color=PALETTE["tol"], ls=":", lw=1.0)
    ax1.fill_between([0.015, 2.0], 0.05, 25.0, color=PALETTE["det_fill"], alpha=0.22, zorder=0)
    ax1.text(0.30, 1.4, r"$25\,\mathrm{bps}$ tolerance", color=PALETTE["det"], fontsize=7.4)

    # Mark Spotlight S7
    sp7 = [s for s in e2["spotlights"] if s["id"] == "S7_cal_spx_full"][0]
    ax1.scatter([sp7["eta_eff"]], [sp7["det_iv_width_bps"]], color=PALETTE["jensen"], s=36, zorder=6, edgecolor="white", lw=0.6)
    ax1.annotate("S7 (cal. SPX)", xy=(sp7["eta_eff"], sp7["det_iv_width_bps"]), xytext=(0.14, 6800),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["jensen"], lw=0.75), fontsize=7.4, color=PALETTE["jensen"])

    ax1.set_xlim(0.018, 1.9)
    ax1.set_ylim(0.1, 12000)
    ax1.set_xlabel(r"Dimensionless vol-of-vol $\eta T^H$")
    ax1.set_ylabel("176-bit Arb ATM IV width (bps)")
    style_axes(ax1, "both")
    panel_title(ax1, "a", r"Arb width scaling ($H=0.10$, $T=0.25$)")

    # Panel (b): Exact 25-bps and 50-bps phase boundary across H
    bnds = e2["boundaries"]
    Hs = np.array([r["H"] for r in bnds])
    b_r0_25 = np.array([r["eta_eff_star_rho0_2m_25bps"] for r in bnds])
    b_r0_50 = np.array([r["eta_eff_star_rho0_2m_50bps"] for r in bnds])
    b_r5_25 = np.array([r["eta_eff_star_rho_m05_k0_25bps"] for r in bnds])
    b_r9_25 = np.array([r["eta_eff_star_rho_m09_k0_25bps"] for r in bnds])
    b_r9_50 = np.array([r["eta_eff_star_rho_m09_k0_50bps"] for r in bnds])

    ax2.semilogy(Hs, b_r0_25, color=PALETTE["det"], lw=1.8, marker="o", ms=4.2)
    ax2.semilogy(Hs, b_r0_50, color=PALETTE["det"], lw=1.0, ls="--", alpha=0.65)
    ax2.semilogy(Hs, b_r5_25, color=PALETTE["markov"], lw=1.35, marker="s", ms=3.8)
    ax2.semilogy(Hs, b_r9_25, color=PALETTE["jensen"], lw=1.5, marker="^", ms=4.0)
    ax2.semilogy(Hs, b_r9_50, color=PALETTE["jensen"], lw=1.0, ls="--", alpha=0.65)
    ax2.fill_between(Hs, b_r9_25, b_r0_25, color=PALETTE["tol_fill"], alpha=0.35, zorder=0)

    anchor_h = 0.25
    # Semilog line segments interpolate in log ordinate in display coordinates.
    lower, upper = (float(np.exp(np.interp(anchor_h, Hs, np.log(y)))) for y in (b_r9_25, b_r0_25))
    ax2.annotate("", xy=(anchor_h, lower), xytext=(anchor_h, upper),
                 arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, color=PALETTE["jensen"], lw=1.0))
    ratios = [Fraction(a) / Fraction(b) for a, b in zip(b_r0_25, b_r9_25)]
    c_min = math.floor(min(ratios))
    c_max = math.ceil(max(ratios))
    ax2.text(0.265, 0.036, f"{c_min}$\\times$\u2013{c_max}$\\times$ contraction\n(leverage $\\times$ 2nd order)", color=PALETTE["jensen"], fontsize=7.4, va="center")
    ax2.text(0.06, 0.45, r"$50\,\mathrm{bps}$ (dashed)", color=PALETTE["muted"], fontsize=7.4)
    ax2.text(0.06, 0.145, r"$25\,\mathrm{bps}$ (solid)", color=PALETTE["det"], fontsize=7.4)

    ax2.set_xlim(0.03, 0.47)
    ax2.set_ylim(0.0025, 0.8)
    ax2.set_xlabel(r"Hurst exponent $H$")
    ax2.set_ylabel(r"Certifiable boundary $\eta_{\mathrm{eff}}^* = (\eta T^H)^*$")
    style_axes(ax2, "both")
    panel_title(ax2, "b", r"Certified $25$ and $50\,\mathrm{bps}$ boundaries")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=7.4)
    return save_figure(fig, FIG_DIR, "fig_e2_phase_map")


# -----------------------------------------------------------------------------
# Figure 3: E3 176-bit Arb Kernel Audit (Markovian & BLP Hybrid)
# -----------------------------------------------------------------------------
def build_fig_e3():
    e3 = load_json("e3_kernel_audit.json")
    mk = e3["markov_audit"]
    blp = e3["blp_hybrid_audit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.80), gridspec_kw={"wspace": 0.30})

    # Panel (a): Markovian L^2 (solid) and rigorous Arb L^1 (dashed) across H
    for H_val, col, mk_sh in [(0.07, H_COLORS["0.07"], "^"), (0.10, H_COLORS["0.10"], "o"), (0.30, H_COLORS["0.30"], "s")]:
        rows = sorted([r for r in mk if abs(r["H"] - H_val) < 1e-9], key=lambda x: x["M_factors"])
        Ms = [r["M_factors"] for r in rows]
        l2s = [r["kernel_L2_hi"] for r in rows]
        l1s = [r["kernel_L1_hi"] for r in rows]
        ax1.semilogy(Ms, l2s, color=col, lw=1.6, marker=mk_sh, ms=4.2, label=f"$H={H_val:.2f}$ ($L^2$)")
        ax1.semilogy(Ms, l1s, color=col, lw=1.0, ls="--", marker=mk_sh, ms=3.0, alpha=0.7)
    ax1.plot([], [], color=PALETTE["muted"], ls="--", lw=1.0, label=r"Arb $L^1$ (dashed)")
    ax1.set_xticks([2, 4, 6, 8, 12, 16])
    ax1.set_xlabel(r"Markovian factors $M$ (unclipped L-BFGS-B)")
    ax1.set_ylabel(r"176-bit Arb kernel error")
    ax1.set_ylim(1e-5, 0.8)
    style_axes(ax1, "both")
    panel_title(ax1, "a", "Bayer\u2013Breneis kernel $L^2$ and $L^1$ decay")

    # Panel (b): H=0.10 Arb certified price gap upper bound vs BLP Hybrid (kappa) & empirical Markovian error
    rows10 = sorted([r for r in mk if abs(r["H"] - 0.10) < 1e-9], key=lambda x: x["M_factors"])
    blp10 = sorted([r for r in blp if abs(r["H"] - 0.10) < 1e-9 and r["n"] == 256], key=lambda x: x["kappa"])
    Ms10 = [r["M_factors"] for r in rows10]
    gaps_mk = [r["arb_price_gap_upper_atm"] for r in rows10]
    kaps = [r["kappa"] for r in blp10]
    gaps_blp = [r["arb_price_gap_upper_atm"] for r in blp10]

    # Map x-axis to index / parameter cleanly using twin scale or shared integer complexity axis
    ax2.semilogy(Ms10, gaps_mk, color=PALETTE["markov"], lw=1.6, marker="o", ms=4.5, label=r"BB2023 Arb gap vs $M$")
    ax2.semilogy(kaps, gaps_blp, color=PALETTE["mp2018"], lw=1.6, marker="s", ms=4.2, label=r"BLP Hybrid ($n=256$) vs $\kappa$")
    # Reference Bridge ATM sampling+bias price half-width
    ref_half = e3["ref_H010_bridge"]["sampling_half_price"][2] + e3["ref_H010_bridge"]["bias_envelope_price"][2]
    ax2.axhline(ref_half, color=PALETTE["roughcert"], ls=":", lw=1.2, label=r"Bridge ATM half-width")
    ax2.set_xticks([1, 2, 4, 6, 8, 12, 16])
    ax2.set_xlabel(r"Markovian factors $M$ or BLP near-field lags $\kappa$")
    ax2.set_ylabel("176-bit Arb ATM price-gap bound")
    ax2.set_ylim(4.0e-5, 0.11)
    style_axes(ax2, "both")
    panel_title(ax2, "b", r"Arb continuous-time price gap ($H=0.10$)")
    ax2.legend(loc="upper right", fontsize=7.4, handlelength=1.5, framealpha=0.95)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=7.4)
    return save_figure(fig, FIG_DIR, "fig_e3_kernel_audit")


# -----------------------------------------------------------------------------
# Figure 4: E4 Regime A Dot Table + Full Smile Comparison
# -----------------------------------------------------------------------------
def build_fig_e4_dot():
    e4 = load_json("e4_benchmarks.json")
    regA = e4["regime_A"]
    refA = regA["ref_bridge"]
    strikes = np.array(regA["params"]["strikes"])
    e_map = {e["method"]: e for e in regA["entries"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.55), gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.30})
    fig.subplots_adjust(left=0.215, right=0.995)  # keep the long method labels inside the canvas (no print shrink)

    # Panel (a): Dot-table of ATM IV error intervals [d_min, d_max] (bps)
    ordered = [
        ("RoughCert-SPX-CE(n=256)", r"$\mathsf{RC\text{-}CE}$ ($n=256$)", PALETTE["roughcert"]),
        ("RoughCert-SPX-CE(n=128)", r"$\mathsf{RC\text{-}CE}$ ($n=128$)", PALETTE["roughcert"]),
        ("RoughCert-SPX-CE-Jensen(n=256)", r"Jensen-$J$ ($n=256$)", PALETTE["jensen"]),
        ("BHT2020-BridgeRQMC(n=256)", r"BHT2020 ($n=256$)", PALETTE["bht"]),
        ("BHT2020-Richardson(64,128)", r"BHT2020 Rich. (diag.)", PALETTE["bht"]),
        ("MP2018-Hybrid(k=2,n=256)", r"MP2018 ($\kappa=2,256$)", PALETTE["mp2018"]),
        ("MP2018-Hybrid(k=1,n=128)", r"MP2018 ($\kappa=1,128$)", PALETTE["mp2018"]),
        ("BayerBreneis2023(M=12,n=256)", r"BB2023 ($M=12,256$)", PALETTE["markov"]),
        ("BayerBreneis2023(M=8,n=128)", r"BB2023 ($M=8,128$)", PALETTE["markov"]),
        ("HJM2024-rDonsker(vm,n=256)", r"HJMS2024 (vm, $256$)", PALETTE["rdonsker"]),
        ("HJM2024-rDonsker(b*,n=256)", r"BLP cell mean ($b_k^*$, $256$)", PALETTE["rdonsker_b"]),
        ("BergomiGuyon2012-O2", r"BG2012 Order-2 (diag.)", PALETTE["bg2012"]),
    ]
    ys = np.arange(len(ordered))[::-1]
    for y, (key, lbl, col) in zip(ys, ordered):
        lo, hi = e_map[key]["atm_iv_err_bps"]
        hi_plot = min(hi, 64.0)
        lo_plot = min(lo, 64.0)
        ax1.plot([lo_plot, hi_plot], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=3)
        if lo == 0.0:
            # Open circle at hi for "at most hi" (inside enclosure!)
            ax1.scatter([hi_plot], [y], facecolor="white", edgecolor=col, s=26, lw=1.2, zorder=4)
        else:
            ax1.scatter([lo_plot, hi_plot], [y, y], color=col, s=20, zorder=4)
        if hi > 64.0:
            lo_f = math.floor(lo)
            hi_c = math.ceil(hi)
            ax1.text(63.0, y, f"{lo_f}\u2013{hi_c}", ha="right", va="center", fontsize=7.4, color=col,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    ax1.axvline(10.0, color=PALETTE["roughcert"], ls=":", lw=0.9)
    ax1.axvline(25.0, color=PALETTE["tol"], ls="--", lw=0.9)
    ax1.text(10.8, len(ordered) - 0.35, "10 bps", fontsize=7.4, color=PALETTE["roughcert"], va="bottom")
    ax1.text(25.8, len(ordered) - 0.35, "25 bps", fontsize=7.4, color=PALETTE["tol"], va="bottom")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([x[1] for x in ordered], fontsize=7.4)
    ax1.set_xlim(-1.5, 65.0)
    ax1.set_ylim(-0.7, len(ordered) + 0.30)
    ax1.set_xlabel("ATM IV error interval vs continuum enclosure (bps)")
    style_axes(ax1, "x")
    panel_title(ax1, "a", "Regime A ATM error intervals")

    # Panel (b): Smile deviation relative to Bridge midpoint across strikes
    mid_iv = np.array(refA["iv_mid"])
    band_lo = 1e4 * (np.array(refA["iv_continuum_lo"]) - mid_iv)
    band_hi = 1e4 * (np.array(refA["iv_continuum_hi"]) - mid_iv)
    ax2.fill_between(strikes, band_lo, band_hi, color=PALETTE["roughcert_fill"], alpha=0.28, label="Bridge continuum band")
    ax2.axhline(0.0, color=PALETTE["roughcert"], lw=0.9, ls=":")

    curves_b = [
        ("RoughCert-SPX-CE(n=256)", r"$\mathsf{RC\text{-}CE}$ ($n=256$)", PALETTE["roughcert"], "o", "-", 1.6),
        ("MP2018-Hybrid(k=2,n=256)", r"MP2018 ($\kappa=2,256$)", PALETTE["mp2018"], "s", "-", 1.2),
        ("BayerBreneis2023(M=12,n=256)", r"BB2023 ($M=12,256$)", PALETTE["markov"], "D", "-", 1.2),
        ("HJM2024-rDonsker(vm,n=256)", r"HJMS2024 (vm, $256$)", PALETTE["rdonsker"], "v", "--", 1.2),
        ("HJM2024-rDonsker(b*,n=256)", r"BLP cell mean ($b_k^*$, $256$)", PALETTE["rdonsker_b"], "^", ":", 1.2),
        ("RoughCert-SPX-CE-Jensen(n=256)", r"Jensen-$J$ ($256$)", PALETTE["jensen"], "x", "--", 1.3),
    ]
    for key, lbl, col, mk, ls, lw in curves_b:
        dev = 1e4 * (np.array(e_map[key]["iv_mid"]) - mid_iv)
        ax2.plot(strikes, dev, color=col, ls=ls, lw=lw, marker=mk, ms=3.8, label=lbl)

    ax2.set_xlabel(r"Strike $K$")
    ax2.set_ylabel("IV deviation from Bridge midpoint (bps)")
    ax2.set_ylim(-135, 52)
    style_axes(ax2, "y")
    panel_title(ax2, "b", "Smile deviations")
    ax2.legend(loc="lower left", fontsize=7.4, ncol=1, handlelength=1.6, borderaxespad=0.3, labelspacing=0.28)

    return save_figure(fig, FIG_DIR, "fig_e4_dot_table")


# -----------------------------------------------------------------------------
# Figure 5: E4 Work-Precision Curves (Amortised Solve vs Cold-Start)
# -----------------------------------------------------------------------------
def build_fig_e4_wp():
    e4 = load_json("e4_benchmarks.json")
    regA = e4["regime_A"]
    e_map = {e["method"]: e for e in regA["entries"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.80), gridspec_kw={"wspace": 0.34})

    families = [
        ("RoughCert-SPX-CE", [f"RoughCert-SPX-CE(n={n})" for n in (32, 64, 128, 256)], PALETTE["roughcert"], "o", "-", 1.8),
        ("BHT2020 Bridge-RQMC", [f"BHT2020-BridgeRQMC(n={n})" for n in (32, 64, 128, 256)], PALETTE["bht"], "s", "--", 1.3),
        (r"MP2018 Hybrid ($\kappa=2$)", [f"MP2018-Hybrid(k=2,n={n})" for n in (32, 64, 128, 256)], PALETTE["mp2018"], "D", "-", 1.3),
        (r"BB2023 Markov ($n=128$)", [f"BayerBreneis2023(M={m},n=128)" for m in (4, 8, 12, 16)], PALETTE["markov"], "^", "-", 1.3),
        ("HJMS2024 rDonsker (vm)", [f"HJM2024-rDonsker(vm,n={n})" for n in (32, 64, 128, 256)], PALETTE["rdonsker"], "v", ":", 1.3),
    ]

    for lbl, keys, col, mk, ls, lw in families:
        t_solve = [e_map[k]["solve_s"] for k in keys]
        t_cold = [e_map[k]["cold_setup_s"] + e_map[k]["solve_s"] for k in keys]
        atm_hi = [e_map[k]["atm_iv_err_bps"][1] for k in keys]
        max_hi = [e_map[k]["max_smile_err_hi_bps"] for k in keys]
        ax1.loglog(t_solve, atm_hi, color=col, ls=ls, lw=lw, marker=mk, ms=4.5, label=lbl)
        ax2.loglog(t_cold, max_hi, color=col, ls=ls, lw=lw, marker=mk, ms=4.5)

    # Annotate Reference ATM midpoint floor (W/2) and full width (W) on panel (a)
    ref_atm_w = regA["ref_bridge"]["iv_continuum_width_bps"][4]
    ref_half_w = 0.5 * ref_atm_w
    ref_half_ceil = dround(ref_half_w, 1, "ceil")
    ref_full_ceil = dround(ref_atm_w, 1, "ceil")
    ax1.axhline(ref_half_w, color=PALETTE["roughcert"], ls=":", lw=1.0)
    ax1.text(0.24, ref_half_w * 0.87, f"Ref. midpoint floor ({ref_half_ceil:.1f} bps)", fontsize=7.4, color=PALETTE["roughcert"])
    ax1.axhline(ref_atm_w, color=PALETTE["muted"], ls="--", lw=0.8, alpha=0.75)
    ax1.text(0.24, ref_atm_w * 1.06, f"Ref. full width ({ref_full_ceil:.1f} bps)", fontsize=7.4, color=PALETTE["muted"])
    ax1.set_ylim(5.4, 42.0)

    # Endpoint labels on panel (b) for grid n and Markovian factors M
    ce32 = e_map["RoughCert-SPX-CE(n=32)"]
    ce256 = e_map["RoughCert-SPX-CE(n=256)"]
    bb4 = e_map["BayerBreneis2023(M=4,n=128)"]
    bb16 = e_map["BayerBreneis2023(M=16,n=128)"]
    ax2.annotate(r"$n=32$", xy=(ce32["cold_setup_s"] + ce32["solve_s"], ce32["max_smile_err_hi_bps"]),
                 xytext=(1.05, 160.0), fontsize=7.4, color=PALETTE["roughcert"], arrowprops=dict(arrowstyle="-", color=PALETTE["roughcert"], lw=0.6, shrinkA=1.0, shrinkB=2.5))
    ax2.annotate(r"$n=256$", xy=(ce256["cold_setup_s"] + ce256["solve_s"], ce256["max_smile_err_hi_bps"]),
                 xytext=(2.4, 48.5), fontsize=7.4, color=PALETTE["roughcert"], arrowprops=dict(arrowstyle="-", color=PALETTE["roughcert"], lw=0.6, shrinkA=1.0, shrinkB=2.5))
    ax2.annotate(r"$M=4$", xy=(bb4["cold_setup_s"] + bb4["solve_s"], bb4["max_smile_err_hi_bps"]),
                 xytext=(0.93, 68.0), fontsize=7.4, color=PALETTE["markov"], arrowprops=dict(arrowstyle="-", color=PALETTE["markov"], lw=0.6, shrinkA=1.0, shrinkB=2.5))
    ax2.annotate(r"$M=16$", xy=(bb16["cold_setup_s"] + bb16["solve_s"], bb16["max_smile_err_hi_bps"]),
                 xytext=(3.3, 97.0), fontsize=7.4, color=PALETTE["markov"], arrowprops=dict(arrowstyle="-", color=PALETTE["markov"], lw=0.6, shrinkA=1.0, shrinkB=2.5))

    plain_log_ticks(ax1, xt=[0.25, 0.5, 1, 2, 5], yt=[6, 8, 10, 15, 20, 30, 40])
    plain_log_ticks(ax2, xt=[0.25, 0.5, 1, 2, 5], yt=[50, 60, 80, 100, 150])
    ax1.set_xlabel("Amortised solve time (s, single thread)")
    ax1.set_ylabel("ATM IV error upper bound (bps)")
    style_axes(ax1, "both")
    panel_title(ax1, "a", "Work\u2013precision: amortised solve (ATM)")

    ax2.set_xlabel("Cold start: setup + solve (s)")
    ax2.set_ylabel("Max smile IV error upper bound (bps)")
    style_axes(ax2, "both")
    panel_title(ax2, "b", "Work\u2013precision: cold start (full smile)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=7.4)
    return save_figure(fig, FIG_DIR, "fig_e4_work_precision")


# -----------------------------------------------------------------------------
# Figure 6: E5 FSW (2025) Weak-Rate Phase Transition & Dual-Channel Crossover
# -----------------------------------------------------------------------------
def build_fig_e5():
    e5 = load_json("e5_weak_rates.json")
    rows = e5["results_by_H"]
    Hs = np.array([r["H"] for r in rows])

    b_lp = [r["methods"]["leftpoint"]["call106_slope"]["beta_hat"] for r in rows]
    b_ce = [r["methods"]["ce_i_only"]["call106_slope"]["beta_hat"] for r in rows]
    b_jen = [r["methods"]["ce_adapted"]["call106_slope"]["beta_hat"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.85), gridspec_kw={"wspace": 0.34})

    # Panel (a): Measured weak rate (K=1.06) vs FSW theory & 2H pitfall curve
    h_dense = np.linspace(0.03, 0.43, 200)
    ax1.plot(h_dense, np.minimum(3.0 * h_dense + 0.5, 1.0), color=PALETTE["ink"], lw=1.4, ls="--", label=r"FSW theory $\min(3H+\frac{1}{2},1)$")
    ax1.plot(h_dense, 2.0 * h_dense, color=PALETTE["muted"], lw=1.0, ls=":", label=r"Deficit order $2H$")
    ax1.axvline(1.0 / 6.0, color=PALETTE["muted"], lw=0.85, ls="-.")
    ax1.text(1.0 / 6.0 + 0.008, 0.16, r"$H=\frac{1}{6}$", fontsize=7.4, color=PALETTE["muted"])

    ax1.plot(Hs, b_ce, color=PALETTE["roughcert"], lw=1.7, marker="o", ms=5.0, label=r"$\mathsf{RoughCert\text{-}SPX\text{-}CE}$")
    ax1.plot(Hs, b_lp, color=PALETTE["bht"], lw=1.3, marker="s", ms=4.2, label="Left-point")
    ax1.plot(Hs, b_jen, color=PALETTE["jensen"], lw=1.3, marker="^", ms=4.5, label=r"Jensen-$J$ ablation")

    ax1.set_xlim(0.03, 0.43)
    ax1.set_ylim(0.05, 1.12)
    ax1.set_xlabel(r"Hurst exponent $H$")
    ax1.set_ylabel(r"Measured weak rate $\hat{\beta}$ ($K=1.06$)")
    style_axes(ax1, "both")
    panel_title(ax1, "a", "Weak rates across $H$")

    # Panel (b): Coupled K=1.06 increment magnitude |Delta_n| vs n at H=0.10 (showing crossover!)
    r10 = [r for r in rows if abs(r["H"] - 0.10) < 1e-9][0]
    levels_c = np.array(e5["params"]["levels"][:-1], dtype=float)
    for m_key, col, mk, lw in [
        ("ce_i_only", PALETTE["roughcert"], "o", 1.7),
        ("leftpoint", PALETTE["bht"], "s", 1.3),
        ("ce_adapted", PALETTE["jensen"], "^", 1.3),
    ]:
        inc = np.abs(np.array(r10["methods"][m_key]["incr_means"])[:, 2])
        se = np.array(r10["methods"][m_key]["incr_ses"])[:, 2]
        pos = inc > 0
        ax2.errorbar(levels_c[pos], inc[pos], yerr=1.96 * se[pos], color=col, lw=lw, marker=mk, ms=4.2, capsize=2.2)

    n_line = np.array([16.0, 256.0])
    ax2.loglog(n_line, 6.5e-4 * (n_line / 16.0) ** (-0.8), color=PALETTE["ink"], ls="--", lw=0.95)
    ax2.text(42.0, 3.05e-4, r"$\propto n^{-0.8}$", fontsize=7.4, color=PALETTE["ink"])
    ax2.loglog(n_line, 2.7e-4 * (n_line / 16.0) ** (-0.2), color=PALETTE["muted"], ls=":", lw=0.95)
    ax2.text(122.0, 1.98e-4, r"$\propto n^{-0.2}$", fontsize=7.4, color=PALETTE["muted"])
    inc_ce = np.abs(np.array(r10["methods"]["ce_i_only"]["incr_means"])[:, 2])
    inc_j = np.abs(np.array(r10["methods"]["ce_adapted"]["incr_means"])[:, 2])
    i_x = next(i for i in range(len(levels_c) - 1) if (inc_ce[i] - inc_j[i]) * (inc_ce[i + 1] - inc_j[i + 1]) < 0)
    d0, d1 = np.log(inc_ce[i_x] / inc_j[i_x]), np.log(inc_ce[i_x + 1] / inc_j[i_x + 1])
    t_x = d0 / (d0 - d1)  # log-log interpolation of the first sign change of |CE| - |Jensen|
    n_x, y_x = levels_c[i_x] * 2.0 ** t_x, inc_j[i_x] * (inc_j[i_x + 1] / inc_j[i_x]) ** t_x
    ax2.annotate("Crossover", xy=(n_x, y_x), xytext=(26.0, 5.2e-5),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["roughcert"], lw=0.8), fontsize=7.4, color=PALETTE["roughcert"])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(levels_c)
    ax2.set_xticklabels([str(int(n)) for n in levels_c])
    ax2.minorticks_off()
    plain_log_ticks(ax2, yt=[3e-5, 1e-4, 3e-4], yfmt=lambda v, _: tex_sci(v))
    ax2.set_xlabel(r"Coarse grid $n$ ($n\to 2n$)")
    ax2.set_ylabel(r"Call-wing increment $|\bar{\Delta}_n|$ ($K=1.06$)")
    style_axes(ax2, "both")
    panel_title(ax2, "b", r"Increment crossover ($K=1.06$)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=7.4)
    return save_figure(fig, FIG_DIR, "fig_e5_weak_rates")


# -----------------------------------------------------------------------------
# Figure 7: E6 Certified Joint SPX-VIX Audit (Pure vs Mixed rBergomi)
# -----------------------------------------------------------------------------
def build_fig_e6():
    e6 = load_json("e6_joint_spx_vix.json")
    vix_pure = e6["paper1_vix_smiles"]["pure_bdg6"]
    vix_mix = e6["paper1_vix_smiles"]["mixed_scen1"]
    rows = {r["model_id"]: r for r in e6["joint_rows"]}
    pure_spx = rows["Pure_BDG6_rho-0.85"]["spx_bridge"]
    mix_spx = rows["Mixed_Scen1_rho-0.85"]["spx_bridge"]
    strikes = np.array(e6["strikes"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.85), gridspec_kw={"wspace": 0.34})

    # Panel (a): 169-bit Arb VIX IV smiles from Paper 1
    k_p = np.array([s["log_moneyness"] for s in vix_pure if -0.31 <= s["log_moneyness"] <= 0.31])
    p_lo = np.array([s["iv_L_pct"] for s in vix_pure if -0.31 <= s["log_moneyness"] <= 0.31])
    p_hi = np.array([s["iv_U_pct"] for s in vix_pure if -0.31 <= s["log_moneyness"] <= 0.31])

    k_m = np.array([s["log_moneyness"] for s in vix_mix])
    m_lo = np.array([s["iv_L_pct"] for s in vix_mix])
    m_hi = np.array([s["iv_U_pct"] for s in vix_mix])

    ax1.fill_between(k_p, p_lo, p_hi, color=PALETTE["markov"], alpha=0.32)
    ax1.plot(k_p, 0.5 * (p_lo + p_hi), color=PALETTE["markov"], lw=1.6, marker="s", ms=4.0, label=r"Pure BDG6 ($a=1.0$, $\eta=2.24$)")
    ax1.fill_between(k_m, m_lo, m_hi, color=PALETTE["mixed"], alpha=0.32)
    ax1.plot(k_m, 0.5 * (m_lo + m_hi), color=PALETTE["mixed"], lw=1.6, marker="o", ms=4.0, label=r"Mixed Scen 1 ($\lambda=0.30$, $\eta_1=3.13$)")

    v_p = rows["Pure_BDG6_rho-0.85"]["paper1_vix_smile_slope_pp"]
    v_m = rows["Mixed_Scen1_rho-0.85"]["paper1_vix_smile_slope_pp"]
    vp_lo = dround(v_p[0], 2, "floor")
    vp_hi = dround(v_p[1], 2, "ceil")
    vm_lo = dround(v_m[0], 2, "floor")
    vm_hi = dround(v_m[1], 2, "ceil")

    ax1.annotate(f"Nearly flat VIX slope\n({vp_lo:+.2f} to {vp_hi:+.2f} pp)", xy=(0.05, 114.3), xytext=(-0.28, 107.0),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["markov"], lw=0.8), fontsize=7.4, color=PALETTE["markov"])
    ax1.annotate(f"Upward VIX smile\n({vm_lo:+.2f} to {vm_hi:+.2f} pp)", xy=(0.10, 97.3), xytext=(-0.28, 100.8),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["mixed"], lw=0.8), fontsize=7.4, color=PALETTE["mixed"])

    ax1.set_xlabel(r"VIX log-moneyness $\log(K_{\mathrm{VIX}}/F_{\mathrm{VIX}})$")
    ax1.set_ylabel("VIX implied volatility (%)")
    ax1.set_ylim(90.5, 117.5)
    style_axes(ax1, "both")
    panel_title(ax1, "a", "RoughCert v5.0.0 Arb VIX smiles")

    # Panel (b): Paired RoughCert-SPX-Bridge SPX smiles at rho = -0.85 (filled: asymp.-env., open: diagnostic)
    s_p_lo = 100.0 * np.array(pure_spx["iv_continuum_lo"])
    s_p_hi = 100.0 * np.array(pure_spx["iv_continuum_hi"])
    s_p_mid = 100.0 * np.array(pure_spx["iv_mid"])
    p_diag = np.array([st == "diagnostic" for st in pure_spx["status_per_strike"]])

    s_m_lo = 100.0 * np.array(mix_spx["iv_continuum_lo"])
    s_m_hi = 100.0 * np.array(mix_spx["iv_continuum_hi"])
    s_m_mid = 100.0 * np.array(mix_spx["iv_mid"])
    m_diag = np.array([st == "diagnostic" for st in mix_spx["status_per_strike"]])

    ax2.fill_between(strikes, s_p_lo, s_p_hi, color=PALETTE["markov"], alpha=0.28)
    ax2.plot(strikes, s_p_mid, color=PALETTE["markov"], lw=1.5)
    ax2.plot(strikes[~p_diag], s_p_mid[~p_diag], linestyle="none", marker="s", ms=4.0, color=PALETTE["markov"])
    if np.any(p_diag):
        ax2.plot(strikes[p_diag], s_p_mid[p_diag], linestyle="none", marker="s", ms=4.3,
                 markerfacecolor="white", markeredgecolor=PALETTE["markov"], markeredgewidth=1.2, zorder=5)

    ax2.fill_between(strikes, s_m_lo, s_m_hi, color=PALETTE["mixed"], alpha=0.28)
    ax2.plot(strikes, s_m_mid, color=PALETTE["mixed"], lw=1.5)
    ax2.plot(strikes[~m_diag], s_m_mid[~m_diag], linestyle="none", marker="o", ms=4.0, color=PALETTE["mixed"])
    if np.any(m_diag):
        ax2.plot(strikes[m_diag], s_m_mid[m_diag], linestyle="none", marker="o", ms=4.3,
                 markerfacecolor="white", markeredgecolor=PALETTE["mixed"], markeredgewidth=1.2, zorder=5)

    sk_p = rows["Pure_BDG6_rho-0.85"]["spx_atm_skew_interval"]
    sk_m = rows["Mixed_Scen1_rho-0.85"]["spx_atm_skew_interval"]
    skp_lo = dround(sk_p[0], 3, "floor")
    skp_hi = dround(sk_p[1], 3, "ceil")
    skm_lo = dround(sk_m[0], 3, "floor")
    skm_hi = dround(sk_m[1], 3, "ceil")
    p_dag = r"^\dagger" if (p_diag[3] or p_diag[5]) else ""
    m_dag = r"^\dagger" if (m_diag[3] or m_diag[5]) else ""
    ax2.text(
        0.955, 25.4,
        f"ATM skew ($\\rho=-0.85$):\nPure: $[{skp_lo:.3f}, {skp_hi:.3f}]{p_dag}$\nMixed: $[{skm_lo:.3f}, {skm_hi:.3f}]{m_dag}$\n($\\dagger$/open: diag. strike)",
        fontsize=7.4, color=PALETTE["ink"],
        bbox=dict(facecolor="white", edgecolor=PALETTE["grid"], lw=0.7, pad=2.8),
    )

    ax2.set_xlabel(r"SPX strike $K$")
    ax2.set_ylabel("SPX implied volatility (%)")
    style_axes(ax2, "both")
    panel_title(ax2, "b", r"Paired RC-Bridge SPX smiles ($\rho=-0.85$)")

    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=7.5)
    return save_figure(fig, FIG_DIR, "fig_e6_joint_spx_vix")


def main():
    apply_style()
    builders = [
        ("fig_e1_exact_rss", build_fig_e1),
        ("fig_e2_phase_map", build_fig_e2),
        ("fig_e3_kernel_audit", build_fig_e3),
        ("fig_e4_dot_table", build_fig_e4_dot),
        ("fig_e4_work_precision", build_fig_e4_wp),
        ("fig_e5_weak_rates", build_fig_e5),
        ("fig_e6_joint_spx_vix", build_fig_e6),
    ]
    for name, fn in builders:
        paths = fn()
        print(f"Built {name:24s} -> {[str(p.relative_to(ROOT)) for p in paths]}")


if __name__ == "__main__":
    main()
