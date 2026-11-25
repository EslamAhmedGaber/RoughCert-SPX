"""Single-source-of-truth LaTeX macro and table generator for RoughCert-SPX (v2).

Every number printed in the manuscript or in the supplement is produced here from the archived JSON
results under an explicit, recorded rounding rule:

  floor    lower bounds, lower ends of certified ranges, certified boundary points, "at least" and
           "narrower by a factor" statements;
  ceil     upper bounds, widths, error magnitudes, and thresholds that a quantity must exceed;
  nearest  point estimates (statistical estimates, timings, exact-law point values).

Directed rounding is applied to the exact binary value of each IEEE double with decimal.Decimal
(ROUND_FLOOR / ROUND_CEILING), so no nudging tolerance can round inward.

Outputs
  manuscript/tables/spx_numbers.tex           \\newcommand macros (main text and supplement)
  manuscript/tables/spx_numbers_rules.json    raw value, rule, digits, printed string and source per macro
  manuscript/tables/tab_*.tex                 main-text booktabs tables (6)
  manuscript/supplement/tables/tab_s*.tex     supplementary tables (5)

The script fails loudly (non-zero exit) on any missing archive, non-finite value, duplicate macro,
or inconsistent provenance record.
"""
from __future__ import annotations

import json
import math
import re
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, getcontext
from pathlib import Path

getcontext().prec = 80

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "manuscript" / "tables"
SOUT = ROOT / "manuscript" / "supplement" / "tables"
MODES = {"floor": ROUND_FLOOR, "ceil": ROUND_CEILING, "nearest": ROUND_HALF_EVEN}


def fatal(msg: str) -> None:
    sys.exit(f"FATAL: {msg}")


def load_json(name: str) -> dict:
    p = RES / name
    if not p.exists():
        fatal(f"missing required result archive {p.name}")
    return json.loads(p.read_text(encoding="utf-8"))


def finite(val, label: str) -> float:
    if val is None or not math.isfinite(float(val)):
        fatal(f"non-finite value for {label}: {val}")
    return float(val)


# ----------------------------------------------------------------------------------------------
# Directed rounding on the exact binary value
# ----------------------------------------------------------------------------------------------
def dround(val, nd: int, rule: str) -> Decimal:
    if rule not in MODES:
        fatal(f"unknown rounding rule {rule}")
    d = Decimal(finite(val, "dround"))
    r = d.quantize(Decimal(1).scaleb(-nd), rounding=MODES[rule])
    return abs(r) if r.is_zero() else r


def dec(val, nd: int, rule: str) -> str:
    return format(dround(val, nd, rule), "f")


def dec_signed(val, nd: int, rule: str) -> str:
    r = dround(val, nd, rule)
    s = format(r, "f")
    return "+" + s if r > 0 else s


def sci(val, nd: int, rule: str) -> str:
    """Scientific notation with a directed mantissa; plain decimals when the exponent is small."""
    v = finite(val, "sci")
    if v == 0.0:
        return "0"
    d = Decimal(v)
    e = d.copy_abs().adjusted()
    if -2 <= e <= 2:
        return dec(v, nd, rule)
    m = d.scaleb(-e).quantize(Decimal(1).scaleb(-nd), rounding=MODES[rule])
    if m.copy_abs() >= 10:
        e += 1
        m = d.scaleb(-e).quantize(Decimal(1).scaleb(-nd), rounding=MODES[rule])
    return f"{format(m, 'f')}\\times 10^{{{e}}}"


def threshold_tex(value) -> str:
    """Print the stated decision threshold without rounding away significant digits."""
    d = Decimal(str(value)).normalize()
    if d.is_zero():
        return "0"
    exponent = d.adjusted()
    if -2 <= exponent <= 2:
        return format(d, "f")
    return f"{d.scaleb(-exponent):f}\\times 10^{{{exponent}}}"


def int_tex(n: int) -> str:
    s = f"{int(n):,}"
    return s.replace(",", "{,}")


def interval(pair, nd: int) -> str:
    return f"$[{dec(pair[0], nd, 'floor')},\\,{dec(pair[1], nd, 'ceil')}]$"


def short_status(s: str) -> str:
    for key, lab in (("Arb", "Arb-cert."), ("asymptotic", "Asymp.-env."), ("simulated-RQMC", "Sim.-RQMC"),
                     ("simulated-MC", "Sim.-MC"), ("diagnostic", "Diagnostic")):
        if s.startswith(key):
            return lab
    return s


# ----------------------------------------------------------------------------------------------
# Macro registry
# ----------------------------------------------------------------------------------------------
REG: dict[str, dict] = {}


def put(name: str, printed: str, raw, rule: str, nd, src: str) -> str:
    if not re.fullmatch(r"[A-Za-z]+", name):
        fatal(f"illegal macro name {name}")
    if name in REG:
        fatal(f"duplicate macro {name}")
    REG[name] = {"printed": printed, "raw": raw, "rule": rule, "digits": nd, "source": src}
    return printed


def mdec(name, raw, rule, nd, src):
    return put(name, dec(raw, nd, rule), float(raw), rule, nd, src)


def msig(name, raw, rule, nd, src):
    return put(name, dec_signed(raw, nd, rule), float(raw), rule, nd, src)


def msci(name, raw, rule, nd, src):
    return put(name, sci(raw, nd, rule), float(raw), rule, nd, src)


def mint(name, n, src):
    return put(name, int_tex(n), int(n), "exact", 0, src)


def mstr(name, s, src):
    return put(name, s, None, "exact", None, src)


def code_constant(path: Path, pattern: str, label: str) -> str:
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    if not m:
        fatal(f"cannot locate {label} in {path.name}")
    return m.group(1)


def vega(K: float, sigma: float, T: float) -> float:
    s = sigma * math.sqrt(T)
    d1 = -math.log(K) / s + s / 2
    return math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) * math.sqrt(T)


def classify(bridge: dict) -> dict:
    """Theorem 4 case per strike: (i) measured rate, (ii) rate assumed (beta = beta_FSW), (iii) diagnostic."""
    out = {"i": 0, "ii": 0, "iii": 0}
    for s, emp in zip(bridge["status_per_strike"], bridge["emp_rate_per_strike"]):
        if s == "diagnostic":
            out["iii"] += 1
        elif emp is None:
            out["ii"] += 1
        else:
            out["i"] += 1
    return out


def k_label(K: float) -> str:
    s = f"{K:.4f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def table(caption: str, label: str, colspec: str, header: str, rows: list[str], notes: str,
          size: str = "\\footnotesize", sep: str | None = None, env: str = "table", place: str = "t") -> str:
    if place == "t" and label.startswith("tab:s_"):
        place = "!htb"  # supplementary tables: set where referenced (the supplement is not re-typeset by the journal)
    lines = [f"\\begin{{{env}}}[{place}]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    if sep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{sep}}}")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule", *rows, "\\bottomrule",
              "\\end{tabular}", "\\par\\smallskip", f"\\parbox{{\\linewidth}}{{\\footnotesize\\emph{{Notes.}} {notes}}}",
              f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOUT.mkdir(parents=True, exist_ok=True)
    e1 = load_json("e1_exact_laws.json")
    e2 = load_json("e2_det_phase_map.json")
    e3 = load_json("e3_kernel_audit.json")
    e4 = load_json("e4_benchmarks.json")
    e5 = load_json("e5_weak_rates.json")
    e6 = load_json("e6_joint_spx_vix.json")
    cv_ab = load_json("cv_ablation.json")
    dcp = load_json("v2_e2_decomposition.json")
    rfn = load_json("v2_diag_refinement.json")
    pcj = load_json("v2_proof_checks.json")
    icj = load_json("v2_independent_check.json")
    if pcj["n_fail"] or icj["n_fail"]:
        fatal("verification archives report failed checks")

    # ------------------------------------------------------------------ precision and constants
    dps = int(code_constant(ROOT / "src" / "bs_arb.py", r"ARB_DPS\s*=\s*(\d+)", "ARB_DPS"))
    bits = int(round((dps + 1) * math.log2(10)))      # python-flint / mpmath dps -> prec convention
    mint("ArbBits", bits, f"src/bs_arb.py ARB_DPS={dps}")
    safety = float(code_constant(ROOT / "src" / "bridge_rqmc.py", r"safety_factor:\s*float\s*=\s*([0-9.]+)", "safety_factor"))
    mdec("BridgeSafety", safety, "nearest", 1, "src/bridge_rqmc.py safety_factor default")
    lo_e, hi_e = (float(x) for x in re.search(r"lo_e,\s*hi_e\s*=\s*([0-9.e-]+),\s*([0-9.e-]+)",
                  (ROOT / "scripts" / "run_e2_det_phase_map.py").read_text(encoding="utf-8")).groups())
    n_halv = int(code_constant(ROOT / "scripts" / "run_e2_det_phase_map.py", r"for _ in range\((\d+)\)", "bisection count"))
    if f"{n_halv} halvings of [1e-4, 2.5]" not in dcp["status"] or (lo_e, hi_e) != (1e-4, 2.5):
        fatal("bisection provenance mismatch between run_e2_det_phase_map.py and v2_e2_decomposition.json")
    msci("ETwoBisectTol", (hi_e - lo_e) / 2 ** n_halv, "ceil", 1, "run_e2_det_phase_map.py bracket/halvings")
    mint("ETwoBisectSteps", n_halv, "run_e2_det_phase_map.py")
    mint("ETwoGridNH", len(e2["H_grid"]), "e2 H_grid")
    mint("ETwoGridNEta", len(e2["eta_eff_grid"]), "e2 eta_eff_grid")

    # ------------------------------------------------------------------ CV ablation
    mdec("CVIncrVRFMin", cv_ab["incr_vrf_min"], "floor", 1, "cv_ablation incr_vrf_min")
    mdec("CVIncrVRFMax", cv_ab["incr_vrf_max"], "ceil", 1, "cv_ablation incr_vrf_max")
    mdec("CVIncrVRFATM", cv_ab["incr_vrf_atm"], "nearest", 1, "cv_ablation incr_vrf_atm (point estimate)")

    # ------------------------------------------------------------------ E1
    he, ss, rss, c1 = e1["heston_exact"], e1["stein_stein_H05_exact"], e1["rough_stein_stein_H010_exact"], e1["containment_summary"]
    msci("EOneHestonMart", he["martingale_error"], "ceil", 1, "e1 heston martingale_error")
    mdec("EOneHestonATMIV", 100.0 * he["ivs"][3], "nearest", 2, "e1 heston ATM IV (exact-law point value)")
    msci("EOneSSRiccati", ss["max_riccati_diff"], "ceil", 2, "e1 SS max_riccati_diff")
    msci("EOneSSMart", ss["martingale_error"], "ceil", 2, "e1 SS martingale_error")
    msci("EOneRSSRefDiff", rss["max_refinement_price_diff"], "ceil", 2, "e1 RSS M_check->M_op price refinement")
    msci("EOneRSSMart", rss["martingale_error"], "ceil", 2, "e1 RSS martingale_error")
    min_vega = min(vega(K, s, rss["T"]) for K, s in zip(rss["strikes"], rss["ivs"]))
    mdec("EOneRSSRefDiffIV", 1e4 * rss["max_refinement_price_diff"] / min_vega, "ceil", 1,
         "first-order IV equivalent: max refinement price difference / min vega (bps)")
    mint("EOneRSSMop", rss["M_op"], "e1 RSS M_op"); mint("EOneRSSMcheck", rss["M_check"], "e1 RSS M_check")
    lp1, ce1, jen1 = c1["leftpoint"], c1["ce_i_only"], c1["ce_adapted"]
    mint("EOneRSSCeDiag", ce1["n_diagnostic_strikes"], "e1 CE diagnostic count")
    b = lambda d, n: d["iv_bias_by_level"][str(n)]["max_abs_bps"]
    mdec("EOneRSSLPMaxLow", b(lp1, 32), "nearest", 2, "e1 LP max |IV bias| n=32 (estimate)")
    mdec("EOneRSSCeMaxLow", b(ce1, 32), "nearest", 2, "e1 CE max |IV bias| n=32 (estimate)")
    mdec("EOneRSSCeGainLow", b(lp1, 32) / b(ce1, 32), "floor", 2, "ratio LP/CE at n=32 ('reduces by at least')")
    mdec("EOneRSSLPMaxFine", b(lp1, 256), "nearest", 2, "e1 LP n=256")
    mdec("EOneRSSCeMaxFine", b(ce1, 256), "nearest", 2, "e1 CE n=256")
    mdec("EOneRSSJenMaxLow", b(jen1, 32), "nearest", 2, "e1 Jensen n=32")
    mdec("EOneRSSJenMaxFine", b(jen1, 256), "nearest", 2, "e1 Jensen n=256")
    mdec("EOneRSSJenOverLPFine", b(jen1, 256) / b(lp1, 256), "floor", 2, "ratio Jensen/LP at n=256")
    mint("EOneRSSJenDiag", jen1["n_diagnostic_strikes"], "e1 Jensen diagnostic count")

    rows = []
    for key, lab in ((("ce_i_only", r"\RCCE{} (primary)")), ("leftpoint", r"Left-point (FSW baseline)"),
                     ("ce_adapted", r"Jensen-$J$ ablation (\texttt{ce\_adapted})")):
        d = c1[key]; bb = d["iv_bias_by_level"]
        rows.append(f"{lab} & {dec(bb['32']['max_abs_bps'], 2, 'nearest')} & {dec(bb['64']['max_abs_bps'], 2, 'nearest')} "
                    f"& {dec(bb['128']['max_abs_bps'], 2, 'nearest')} & {dec(bb['256']['max_abs_bps'], 2, 'nearest')} "
                    f"& ${dec_signed(bb['256']['atm_bps'], 2, 'nearest')}$ & {dec(d['atm_continuum_width_bps'], 1, 'ceil')} "
                    f"& {sum(d['contained_per_strike'])}/{len(d['contained_per_strike'])} & {d['n_diagnostic_strikes']} \\\\")
    (OUT / "tab_e1_rss.tex").write_text(table(
        "Exact-law validation on correlated rough Stein--Stein ($H=0.10$, $T=0.5$, $\\rho=-0.7$, $7$ strikes in $[0.85,1.15]$).",
        "tab:e1_rss", "@{}lrrrrrrrr@{}",
        "Estimator & \\multicolumn{4}{c}{Max $|\\text{IV bias}|$ (bps)} & ATM$_{256}$ & ATM width & Cont. & Diag. \\\\\n"
        "\\cmidrule(lr){2-5}\n"
        "(coupled RQMC, $32{,}768$ paths) & $n{=}32$ & $n{=}64$ & $n{=}128$ & $n{=}256$ & (bps) & (bps) & (of $7$) & strikes \\\\",
        rows,
        "Reference: Carleman--Fredholm $\\det_3$ Fourier prices with $M=\\EOneRSSMop$ (exact identity plus quadrature; "
        "refinement error at most $\\EOneRSSRefDiffIV\\bps$ in IV). Bias columns are point estimates. ATM width: \\RCBridge{} "
        "asymptotic-envelope statistical(0.95) width at $K=1$. Cont.: reference prices enclosed; Diag.: "
        "Theorem~\\ref{thm:bridge_envelope}(iii) flags.", sep="3.4pt"), encoding="utf-8")

    # ------------------------------------------------------------------ E2
    bnds = e2["boundaries"]
    r0 = [r["eta_eff_star_rho0_2m_25bps"] for r in bnds]
    r9 = [r["eta_eff_star_rho_m09_k0_25bps"] for r in bnds]
    coll = [a / c for a, c in zip(r0, r9)]
    mdec("ETwoRhoZeroMin", min(r0), "floor", 3, "e2 boundary rho=0 two-moment 25 bps (certified point)")
    mdec("ETwoRhoZeroMax", max(r0), "ceil", 3, "range maximum of the rho=0 two-moment 25 bps boundary points")
    mdec("ETwoRhoNineMin", min(r9), "floor", 4, "e2 boundary rho=-0.9 25 bps (certified point)")
    mdec("ETwoRhoNineMax", max(r9), "ceil", 4, "range maximum of the rho=-0.9 25 bps boundary points")
    mdec("ETwoCollapseMin", min(coll), "floor", 0, "ratio of certified boundaries (outer range)")
    mdec("ETwoCollapseMax", max(coll), "ceil", 0, "ratio of certified boundaries (outer range)")
    drows = dcp["rows"]
    if [round(r["H"], 2) for r in drows] != [round(r["H"], 2) for r in bnds]:
        fatal("decomposition H grid differs from E2 boundaries")
    k0z = [r["eta_eff_star_rho0_k0_25bps"] for r in drows]
    so = [r["factor_second_order_25bps"] for r in drows]
    lev = [r["factor_leverage_25bps"] for r in drows]
    for r, a, c in zip(drows, r0, r9):
        if abs(r["factor_total_25bps"] - a / c) > 1e-9 * a / c:
            fatal("decomposition total factor inconsistent with E2 boundaries")
    mdec("ETwoRhoZeroFirstMin", min(k0z), "floor", 3, "v2 decomposition rho=0 first-order (k0) boundary")
    mdec("ETwoRhoZeroFirstMax", max(k0z), "ceil", 3, "range maximum of the rho=0 first-order (k0) boundary points")
    mdec("ETwoSecondOrderMin", min(so), "floor", 1, "v2 decomposition second-order factor (outer range)")
    mdec("ETwoSecondOrderMax", max(so), "ceil", 1, "v2 decomposition second-order factor (outer range)")
    mdec("ETwoLeverageMin", min(lev), "floor", 2, "v2 decomposition leverage factor (outer range)")
    mdec("ETwoLeverageMax", max(lev), "ceil", 2, "v2 decomposition leverage factor (outer range)")

    sp = {s["id"]: s for s in e2["spotlights"]}
    mdec("ETwoSOneDet", sp["S1_mild_uncorr"]["det_iv_width_bps"], "ceil", 2, "e2 S1 width")
    mdec("ETwoSTwoDet", sp["S2_rough_uncorr_mod"]["det_iv_width_bps"], "ceil", 2, "e2 S2 width")
    mdec("ETwoSThreeDet", sp["S3_short_1d_corr"]["det_iv_width_bps"], "ceil", 2, "e2 S3 width")
    mdec("ETwoSFourBridge", sp["S4_mild_rough_corr"]["bridge_iv_width_bps"], "ceil", 2, "e2 S4 bridge width")
    mdec("ETwoSSevenDet", sp["S7_cal_spx_full"]["det_iv_width_bps"], "ceil", 1, "e2 S7 width")
    mdec("ETwoSSevenBridge", sp["S7_cal_spx_full"]["bridge_iv_width_bps"], "ceil", 2, "e2 S7 bridge width")
    mdec("ETwoSSevenRatio", sp["S7_cal_spx_full"]["det_to_bridge_width_ratio"], "floor", 0, "width ratio (wider by at least)")
    i1 = next(c for c in icj["checks"] if c["id"] == "I1")
    ovh = [r["relative_overhead"] for r in i1["rows"]]
    mdec("ETwoArbOverheadMin", 100 * min(ovh), "floor", 1, "I1 Arb width overhead over exact-formula width (%)")
    mdec("ETwoArbOverheadMax", 100 * max(ovh), "ceil", 1, "I1 Arb width overhead over exact-formula width (%)")

    short = {"S1_mild_uncorr": ("S1", "Mild, $\\rho=0$"), "S2_rough_uncorr_mod": ("S2", "Rough, $\\rho=0$"),
             "S3_short_1d_corr": ("S3", "1-day, corr."), "S4_mild_rough_corr": ("S4", "Rough, mild $\\eta$"),
             "S5_moderate_rough": ("S5", "Rough, mod.\\ $\\eta$"), "S6_cal_spx_uncorr": ("S6", "Cal.\\ $\\eta$, $\\rho=0$"),
             "S7_cal_spx_full": ("S7", "Cal.\\ SPX full")}
    rows = []
    for s in e2["spotlights"]:
        code, desc = short[s["id"]]
        t_str = "$1/252$" if s["T"] < 0.01 else f"{s['T']:.2f}"
        dw = s["det_iv_width_bps"]
        det_w = dec(dw, 2, "ceil") if dw < 100 else dec(dw, 1, "ceil")
        gate = "Tier~1 (Arb)" if dw <= 25.0 else "Tier~3 (Bridge)"
        rows.append(f"{code}: {desc} & {s['H']:.2f} & {t_str} & {s['eta']:.2f} & ${s['rho']:.1f}$ & {det_w} "
                    f"& {dec(s['bridge_iv_width_bps'], 2, 'ceil')} & {short_status(s['bridge_status'])} "
                    f"& {'Yes' if s['det_contains_bridge_mid'] else 'No'} & {gate} \\\\")
    (OUT / "tab_e2_spotlights.tex").write_text(table(
        "Spotlight comparison of \\ArbBits-bit Arb \\RCDet{} and \\RCBridge{} ATM implied-volatility widths ($\\xi_0=0.04$, $25\\bps$ desk gate).",
        "tab:e2_spotlights", "@{}lrrrrrrlll@{}",
        "Spotlight & $H$ & $T$ & $\\eta$ & $\\rho$ & Det width & Bridge width & Bridge & Arb & Gate \\\\\n"
        "regime & & (yr) & & & (bps, Arb) & (bps, stat.) & status & cont.? & choice \\\\",
        rows,
        "Det: deterministic Arb-certified ATM IV width (Proposition~\\ref{prop:twomoment} for $\\rho=0$, "
        "Theorem~\\ref{thm:put_lipschitz} for $\\rho<0$). Bridge: asymptotic-envelope statistical(0.95) width "
        "(Theorem~\\ref{thm:bridge_envelope}; levels $32$--$256$, $32{,}768$ RQMC paths). The two widths carry different "
        "guarantees. Arb cont.: the Arb bracket contains the Bridge midpoint.", sep="3.6pt"), encoding="utf-8")

    # ------------------------------------------------------------------ E3
    ref3 = e3["ref_H010_bridge"]
    mdec("EThreeRefATMLo", 100.0 * ref3["iv_continuum_lo"][2], "floor", 2, "e3 ref ATM IV lo (%)")
    mdec("EThreeRefATMHi", 100.0 * ref3["iv_continuum_hi"][2], "ceil", 2, "e3 ref ATM IV hi (%)")
    mdec("EThreeRefATMW", ref3["iv_continuum_width_bps"][2], "ceil", 1, "e3 ref ATM width")
    mdec("EThreeRefSkewLo", e3["ref_H010_skew_interval"][0], "floor", 3, "e3 ref skew lo")
    mdec("EThreeRefSkewHi", e3["ref_H010_skew_interval"][1], "ceil", 3, "e3 ref skew hi")
    mk = lambda H: [r for r in e3["markov_audit"] if abs(r["H"] - H) < 1e-12]
    m10 = {r["M_factors"]: r for r in mk(0.10)}
    mdec("EThreeMkTenMTwoLTwo", m10[2]["kernel_L2_hi"], "ceil", 4, "e3 Markov L2 upper")
    mdec("EThreeMkTenMSixteenLTwo", m10[16]["kernel_L2_hi"], "ceil", 4, "e3 Markov L2 upper")
    mdec("EThreeMkTenMSixteenLOne", m10[16]["kernel_L1_hi"], "ceil", 5, "e3 Markov L1 upper")
    mdec("EThreeMkTenMSixteenGap", m10[16]["arb_price_gap_upper_atm"], "ceil", 5, "e3 Markov price gap upper")
    mdec("EThreeMkTenMSixteenFit", m10[16]["bb_fit_s"], "nearest", 2, "e3 fit time")
    iv_mid3 = 0.5 * (ref3["iv_continuum_lo"][2] + ref3["iv_continuum_hi"][2])
    vg3 = vega(1.0, iv_mid3, e3["params"]["T"])
    mdec("EThreeMkTenMSixteenGapIV", 1e4 * m10[16]["arb_price_gap_upper_atm"] / vg3, "ceil", 0,
         "first-order IV equivalent of the certified M=16 ATM price gap (bps)")
    mdec("EThreeMkTenMTwelveGapIV", 1e4 * m10[12]["arb_price_gap_upper_atm"] / vg3, "ceil", 0,
         "first-order IV equivalent of the certified M=12 ATM price gap (bps)")
    mdec("EThreeMkGapOverEmp", (1e4 * m10[16]["arb_price_gap_upper_atm"] / vg3) / m10[16]["atm_iv_err_interval_bps"][1], "floor", 0,
         "certified M=16 gap (IV equivalent) over empirical ATM error upper end")
    mdec("EThreeMkTenMTwoATMLo", m10[2]["atm_iv_err_interval_bps"][0], "floor", 1, "e3 Markov M=2 ATM err lo (stat.)")
    mdec("EThreeMkTenMSixteenATMHi", m10[16]["atm_iv_err_interval_bps"][1], "ceil", 1, "e3 Markov M=16 ATM err hi (stat.)")
    mdec("EThreeMkSevenMSixteenLTwo", [r for r in mk(0.07) if r["M_factors"] == 16][0]["kernel_L2_hi"], "ceil", 4, "e3 H=0.07")
    mdec("EThreeMkThirtyMSixteenLTwo", [r for r in mk(0.30) if r["M_factors"] == 16][0]["kernel_L2_hi"], "ceil", 5, "e3 H=0.30")
    blp = {r["kappa"]: r for r in e3["blp_hybrid_audit"] if abs(r["H"] - 0.10) < 1e-12 and r["n"] == 256}
    for kap, tag in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
        mdec(f"EThreeBLPKap{tag}IV", blp[kap]["arb_iv_marginal_gap_upper_bps_atm"], "ceil", 1,
             "e3 BLP certified price gap mapped across the statistical reference ATM bracket (conditional)")
    rows_a = [f"BB2023 ($M={r['M_factors']}$) & {dec(r['kernel_L2_hi'], 4, 'ceil')} & {dec(r['kernel_L1_hi'], 5, 'ceil')} "
              f"& {dec(r['E_J_hi'], 5, 'ceil')} & {dec(r['arb_price_gap_upper_atm'], 5, 'ceil')} & {dec(r['bb_fit_s'], 2, 'nearest')} "
              f"& {interval(r['atm_iv_err_interval_bps'], 1)} & ${dec(r['mk_atm_skew'], 4, 'nearest')}$ \\\\" for r in mk(0.10)]
    rows_b = [f"BLP hybrid ($\\kappa={k}$, $n=256$) & -- & -- & {dec(blp[k]['E_J_hi'], 5, 'ceil')} "
              f"& {dec(blp[k]['arb_price_gap_upper_atm'], 5, 'ceil')} & -- "
              f"& $\\le {dec(blp[k]['arb_iv_marginal_gap_upper_bps_atm'], 1, 'ceil')}^{{\\ddagger}}$ & -- \\\\" for k in (1, 2, 4, 8)]
    (OUT / "tab_e3_kernel.tex").write_text(table(
        "\\ArbBits-bit Arb continuous-time kernel audit at $H=0.10$, $T=0.25$, $\\eta=1.9$, $\\rho=-0.9$, $\\xi_0=0.04$.",
        "tab:e3_kernel", "@{}lrrrrrrr@{}",
        "Kernel approximation & $\\|K{-}\\hat{K}\\|_{L^2}$ & $\\|K{-}\\hat{K}\\|_{L^1}$ & $E_J$ upper & Price gap & Fit & ATM IV err & Emp.\\ ATM \\\\\n"
        " & (Arb) & (Arb) & (Arb) & (Arb) & (s) & (bps, stat.) & skew \\\\",
        rows_a + ["\\midrule"] + rows_b,
        "Upper panel: unclipped L-BFGS-B factors of \\citet{BayerBreneis2023}; columns 2--5 are Arb-certified upper bounds; "
        "the ATM error interval ($n=256$) is statistical, against the \\RCBridge{} reference ATM IV "
        "$[\\EThreeRefATMLo,\\EThreeRefATMHi]\\%$. Lower panel: continuous-time hybrid kernel. "
        "$^{\\ddagger}$Certified gap mapped to IV across the statistical reference bracket (conditional; time stepping excluded).", sep="4.0pt"), encoding="utf-8")

    # ------------------------------------------------------------------ E4
    regA, regB, regC = e4["regime_A"], e4["regime_B"], e4["regime_C"]
    for tag, reg, ia in (("A", regA, 4), ("B", regB, 3), ("C", regC, 3)):
        rb = reg["ref_bridge"]
        mdec(f"EFour{tag}RefATMLo", 100.0 * rb["iv_continuum_lo"][ia], "floor", 2, f"e4 {tag} ref ATM lo")
        mdec(f"EFour{tag}RefATMHi", 100.0 * rb["iv_continuum_hi"][ia], "ceil", 2, f"e4 {tag} ref ATM hi")
        mdec(f"EFour{tag}RefATMW", rb["iv_continuum_width_bps"][ia], "ceil", 1, f"e4 {tag} ref ATM width")
        mdec(f"EFour{tag}RefSkewLo", reg["ref_skew_interval"][0], "floor", 3, f"e4 {tag} skew lo")
        mdec(f"EFour{tag}RefSkewHi", reg["ref_skew_interval"][1], "ceil", 3, f"e4 {tag} skew hi")
        if tag in ("A", "B"):
            mint(f"EFour{tag}RefDiag", sum(1 for s in rb["status_per_strike"] if s == "diagnostic"), f"e4 {tag} diag count")
    eA = {e["method"]: e for e in regA["entries"]}
    eB = {e["method"]: e for e in regB["entries"]}
    eC = {e["method"]: e for e in regC["entries"]}
    mdec("EFourACEOneTwoEightATMHi", eA["RoughCert-SPX-CE(n=128)"]["atm_iv_err_bps"][1], "ceil", 1, "e4")
    mdec("EFourACETwoFiveSixATMHi", eA["RoughCert-SPX-CE(n=256)"]["atm_iv_err_bps"][1], "ceil", 1, "e4")
    mdec("EFourACETwoFiveSixSolve", eA["RoughCert-SPX-CE(n=256)"]["solve_s"], "nearest", 2, "e4 timing")
    jen = eA["RoughCert-SPX-CE-Jensen(n=256)"]
    mdec("EFourAJenTwoFiveSixMaxLo", jen["max_smile_err_lo_bps"], "floor", 1, "e4 point-estimate distance")
    U_atm = regA["ref_bridge"]["iv_continuum_hi"][4]
    ci_gap = 1e4 * (jen["iv_ci_lo"][4] - U_atm)
    if ci_gap <= 0:
        fatal("Jensen-J ATM confidence interval no longer separated from the reference envelope")
    mdec("EFourAJenTwoFiveSixATMCIDist", ci_gap, "floor", 1, "distance of Jensen-J 95% CI from reference envelope (bps)")
    mdec("EFourABGMaxLo", eA["BergomiGuyon2012-O2"]["max_smile_err_lo_bps"], "floor", 0, "e4")
    mdec("EFourBBGMaxLo", eB["BergomiGuyon2012-O2"]["max_smile_err_lo_bps"], "floor", 0, "e4")
    mdec("EFourBBGSkewLo", eB["BergomiGuyon2012-O2"]["atm_skew_err_interval"][0], "floor", 3, "e4")
    mdec("EFourARDVmTwoFiveSixMaxHi", eA["HJM2024-rDonsker(vm,n=256)"]["max_smile_err_hi_bps"], "ceil", 1, "e4")
    mdec("EFourARDBsTwoFiveSixMaxLo", eA["HJM2024-rDonsker(b*,n=256)"]["max_smile_err_lo_bps"], "floor", 1, "e4")
    vr256 = float(eA["HJM2024-rDonsker(b*,n=256)"]["terminal_var_ratio"])
    mdec("EFourARDBsVarRatio", vr256, "nearest", 3, "e4 terminal variance ratio (exact kernel sum)")
    mdec("EFourARDBsDeficitPct", 100.0 * (1.0 - vr256), "nearest", 1, "e4")
    mdec("EFourARDBsVarRatioSixtyFour", eA["HJM2024-rDonsker(b*,n=64)"]["terminal_var_ratio"], "nearest", 3, "e4")
    mdec("EFourARDVmVarRatio", max(eA[f"HJM2024-rDonsker(vm,n={n})"]["terminal_var_ratio"] for n in (32, 64, 128, 256)),
         "nearest", 3, "e4 vm terminal variance ratio")
    mdec("EFourABBMTwelveSetup", eA["BayerBreneis2023(M=12,n=256)"]["cold_setup_s"], "nearest", 2, "e4 timing")
    mdec("EFourABBMTwelveSolveTwoFiveSix", eA["BayerBreneis2023(M=12,n=256)"]["solve_s"], "nearest", 2, "e4 timing")
    mdec("EFourAMPTwoOneTwoEightSolve", eA["MP2018-Hybrid(k=2,n=128)"]["solve_s"], "nearest", 2, "e4 timing")
    mdec("EFourAMPTwoTwoFiveSixSolve", eA["MP2018-Hybrid(k=2,n=256)"]["solve_s"], "nearest", 2, "e4 timing")
    mdec("EFourAMPTwoTwoFiveSixATMHi", eA["MP2018-Hybrid(k=2,n=256)"]["atm_iv_err_bps"][1], "ceil", 1, "e4")
    i7 = next(c for c in icj["checks"] if c["id"] == "I7")
    mdec("IndepBGMinLoRefined", i7["value"], "floor", 0, "I7 BG2012 lower error vs refined references (bps)")
    mdec("IndepJenMaxLoRefined", i7["jensen256"][0]["max_smile_err_lo_bps_refined"], "floor", 1, "I7 Jensen n=256 vs refined")

    def samp(e):
        if "diagnostic" in e["status"]:
            return "Diag."
        return "RQMC" if "RQMC" in e["sampling"] else ("MC" if "MC" in e["sampling"] else "Anal.")

    selA = [("RoughCert-SPX-CE(n=128)", r"\RCCE{} ($n=128$)"), ("RoughCert-SPX-CE(n=256)", r"\RCCE{} ($n=256$)"),
            ("RoughCert-SPX-CE-Jensen(n=256)", r"Jensen-$J$ ablation ($n=256$)"),
            ("BHT2020-BridgeRQMC(n=128)", r"BHT2020 Bridge-RQMC ($n=128$)"), ("BHT2020-BridgeRQMC(n=256)", r"BHT2020 Bridge-RQMC ($n=256$)"),
            ("BHT2020-Richardson(64,128)", r"BHT2020 Richardson (64,128)"),
            ("MP2018-Hybrid(k=1,n=128)", r"MP2018 hybrid ($\kappa=1,n=128$)"), ("MP2018-Hybrid(k=2,n=256)", r"MP2018 hybrid ($\kappa=2,n=256$)"),
            ("BayerBreneis2023(M=8,n=128)", r"BB2023 Markov ($M=8,n=128$)"), ("BayerBreneis2023(M=12,n=256)", r"BB2023 Markov ($M=12,n=256$)"),
            ("HJM2024-rDonsker(vm,n=256)", r"HJMS2024 rDonsker (vm, $n=256$)"), ("HJM2024-rDonsker(b*,n=256)", r"rDonsker (BLP $b_k^*$, $n=256$)"),
            ("BergomiGuyon2012-O2", r"BG2012 order-2 expansion")]
    rows = [f"{lab} & {samp(eA[k])} & {dec(eA[k]['cold_setup_s'], 2, 'nearest')} & {dec(eA[k]['solve_s'], 2, 'nearest')} "
            f"& {interval(eA[k]['atm_iv_err_bps'], 1)} & {interval([eA[k]['max_smile_err_lo_bps'], eA[k]['max_smile_err_hi_bps']], 1)} "
            f"& ${dec(eA[k]['atm_skew'], 3, 'nearest')}$ & {interval(eA[k]['atm_skew_err_interval'], 3)} \\\\" for k, lab in selA]
    (OUT / "tab_e4_regimeA.tex").write_text(table(
        "Head-to-head SPX benchmarks in Regime~A ($H=0.10$, $T=0.25$, $\\eta=1.9$, $\\rho=-0.9$, $\\xi_0=0.04$, $8$ strikes in $[0.85,1.12]$).",
        "tab:e4_regimeA", "@{}llrrrrrr@{}",
        "Method & Type & Setup & Solve & ATM IV err & Max smile err & ATM & Skew err \\\\\n"
        "($32{,}768$ paths; 1 thread) & & (s) & (s) & (bps) & (bps) & skew & interval \\\\",
        rows,
        "Errors: $[\\operatorname{dist}(v,[L,U]),\\max(|v-L|,|v-U|)]$ for the point estimate $v$ against the $131{,}072$-path "
        "\\RCBridge{} reference (ATM IV $[\\EFourARefATMLo,\\EFourARefATMHi]\\%$, skew $[\\EFourARefSkewLo,\\EFourARefSkewHi]$); "
        "the sampling error of $v$ is not included. Type: scrambled-Sobol RQMC, pseudo-random MC, analytic or diagnostic. "
        "Simulated methods share common random numbers.", sep="3.6pt"), encoding="utf-8")

    rows = []
    for k, lab in (("RoughCert-SPX-CE(n=128)", r"B: \RCCE{} ($n=128$)"), ("BHT2020-BridgeRQMC(n=128)", r"B: BHT2020 ($n=128$)"),
                   ("MP2018-Hybrid(k=2,n=128)", r"B: MP2018 ($\kappa=2,n=128$)"), ("BayerBreneis2023(M=12,n=128)", r"B: BB2023 ($M=12,n=128$)"),
                   ("HJM2024-rDonsker(vm,n=128)", r"B: HJMS2024 (vm, $n=128$)"), ("BergomiGuyon2012-O2", r"B: BG2012 order-2")):
        e = eB[k]
        rows.append(f"{lab} & {samp(e)} & {dec(e['solve_s'], 2, 'nearest')} & {interval(e['atm_iv_err_bps'], 1)} "
                    f"& {interval([e['max_smile_err_lo_bps'], e['max_smile_err_hi_bps']], 1)} & ${dec(e['atm_skew'], 3, 'nearest')}$ \\\\")
    rows.append("\\midrule")
    for k, lab in (("RoughCert-SPX-CE(n=128)", r"C: \RCCE{} ($n=128$)"), ("BHT2020-BridgeRQMC(n=128)", r"C: BHT2020 ($n=128$)"),
                   ("MP2018-Hybrid(k=2,n=128)", r"C: MP2018 ($\kappa=2,n=128$)")):
        e = eC[k]
        rows.append(f"{lab} & {samp(e)} & {dec(e['solve_s'], 2, 'nearest')} & {interval(e['atm_iv_err_bps'], 1)} "
                    f"& {interval([e['max_smile_err_lo_bps'], e['max_smile_err_hi_bps']], 1)} & ${dec(e['atm_skew'], 3, 'nearest')}$ \\\\")
    (OUT / "tab_e4_regimesBC.tex").write_text(table(
        "Head-to-head benchmarks in Regime~B (ultra-rough one-month, $H=0.07$) and Regime~C (curved $\\xi_0(t)$, $T=0.50$).",
        "tab:e4_regimesBC", "@{}llrrrr@{}",
        "Regime and method & Type & Solve (s) & ATM IV err (bps) & Max smile err (bps) & ATM skew \\\\",
        rows,
        "Regime~B ($H=0.07$, $T=1/12$, $\\eta=1.9$, $\\rho=-0.9$, $\\xi_0=0.235^2$): \\RCBridge{} ATM IV "
        "$[\\EFourBRefATMLo,\\EFourBRefATMHi]\\%$, skew $[\\EFourBRefSkewLo,\\EFourBRefSkewHi]$. Regime~C ($H=0.10$, $T=0.50$, $\\eta=1.9$, "
        "$\\rho=-0.85$, $\\xi_0(t)=0.234^2(1+t)^2$): ATM IV $[\\EFourCRefATMLo,\\EFourCRefATMHi]\\%$, skew $[\\EFourCRefSkewLo,\\EFourCRefSkewHi]$. "
        "Error intervals as in Table~\\ref{tab:e4_regimeA}."), encoding="utf-8")

    # ------------------------------------------------------------------ E5
    e5H = {f"{r['H']:.2f}": r for r in e5["results_by_H"]}
    for hs, tag in (("0.05", "ZeroFive"), ("0.10", "Ten"), ("0.15", "Fifteen"), ("0.25", "TwoFive"), ("0.40", "Forty")):
        for mk_, mt in (("leftpoint", "LP"), ("ce_i_only", "CE"), ("ce_adapted", "Jen")):
            if mt == "Jen" and tag == "Forty":
                continue
            v = e5H[hs]["methods"][mk_]["call106_slope"]["beta_hat"]
            if v is None:
                mstr(f"EFive{mt}H{tag}", "--", "e5 slope unavailable")
            else:
                mdec(f"EFive{mt}H{tag}", v, "nearest", 3, "e5 fitted slope (point estimate)")
    m10_5, m15_5, m05_5 = e5H["0.10"]["methods"], e5H["0.15"]["methods"], e5H["0.05"]["methods"]
    msci("EFiveTenCECoarse", abs(m10_5["ce_i_only"]["incr_means"][0][2]), "nearest", 2, "e5 increment (estimate)")
    msci("EFiveTenCEFine", abs(m10_5["ce_i_only"]["incr_means"][-1][2]), "nearest", 2, "e5 increment (estimate)")
    msci("EFiveTenLPCoarse", abs(m10_5["leftpoint"]["incr_means"][0][2]), "nearest", 2, "e5 increment (estimate)")
    msci("EFiveTenJenCoarse", abs(m10_5["ce_adapted"]["incr_means"][0][2]), "nearest", 2, "e5 increment (estimate)")
    msci("EFiveTenJenFine", abs(m10_5["ce_adapted"]["incr_means"][-1][2]), "nearest", 2, "e5 increment (estimate)")
    mdec("EFiveTenStripCE", m10_5["ce_i_only"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5")
    mdec("EFiveTenStripJen", m10_5["ce_adapted"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5")
    mdec("EFiveATMFifteenCE", m15_5["ce_i_only"]["atm_slope"]["beta_hat"], "nearest", 3, "e5")
    mdec("EFiveATMFifteenJen", m15_5["ce_adapted"]["atm_slope"]["beta_hat"], "nearest", 3, "e5")
    mdec("EFiveStripFifteenCE", m15_5["ce_i_only"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5")
    mdec("EFiveStripFifteenJen", m15_5["ce_adapted"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5")
    msig("EFiveStripCEHZeroFive", m05_5["ce_i_only"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5 H=0.05 (below floor)")
    msig("EFiveStripLPHZeroFive", m05_5["leftpoint"]["strip_mean_slope"]["beta_hat"], "nearest", 3, "e5 H=0.05 (below floor)")
    mdec("EFiveATMCEHZeroFive", m05_5["ce_i_only"]["atm_slope"]["beta_hat"], "nearest", 3, "e5 H=0.05 (below floor)")
    floor_ = pcj["case_ii_rate_floor"]
    for hs, tag in (("0.05", "ZeroFive"), ("0.07", "Seven"), ("0.10", "Ten"), ("0.15", "Fifteen"), ("0.25", "Sixth")):
        mdec(f"BetaMin{tag}", floor_[hs], "ceil", 3, "Case (ii) rate floor log2(1+(2^beta_FSW-1)/safety)")
    c7b = next(c for c in pcj["checks"] if c["id"] == "C7b")
    mdec("EFiveMinMargin", c7b["value"], "floor", 3, "C7b min(beta_hat - beta_min) over H >= 0.10")

    # ------------------------------------------------------------------ E6
    e6m = {r["model_id"]: r for r in e6["joint_rows"]}
    b70, m70, b85, m85 = (e6m[k] for k in ("Pure_BDG6_rho-0.70", "Mixed_Scen1_rho-0.70", "Pure_BDG6_rho-0.85", "Mixed_Scen1_rho-0.85"))
    msig("ESixPureVixSlopeLo", b85["paper1_vix_smile_slope_pp"][0], "floor", 2, "e6 (v5.0.0 Arb)")
    msig("ESixPureVixSlopeHi", b85["paper1_vix_smile_slope_pp"][1], "ceil", 2, "e6 (v5.0.0 Arb)")
    msig("ESixMixVixSlopeLo", m85["paper1_vix_smile_slope_pp"][0], "floor", 2, "e6 (v5.0.0 Arb)")
    msig("ESixMixVixSlopeHi", m85["paper1_vix_smile_slope_pp"][1], "ceil", 2, "e6 (v5.0.0 Arb)")
    mdec("ESixVixLiftLo", m85["paper1_vix_smile_slope_pp"][0] - b85["paper1_vix_smile_slope_pp"][1], "floor", 2,
         "certified lower bound on the mixed-minus-pure VIX slope difference (pp)")
    for tag, r in (("PureSeventy", b70), ("MixSeventy", m70), ("Pure", b85), ("Mix", m85)):
        mdec(f"ESix{tag}SkewLo", r["spx_atm_skew_interval"][0], "floor", 3, "e6 skew lo")
        mdec(f"ESix{tag}SkewHi", r["spx_atm_skew_interval"][1], "ceil", 3, "e6 skew hi")
    mdec("ESixMixCoarseSkew", m85["spx_coarse_lp64_atm_skew"], "nearest", 3, "e6 coarse LP point estimate")
    mint("ESixTotalDiag", sum(r["n_diagnostic_strikes"] for r in e6["joint_rows"]), "e6")
    mint("ESixMixDiagHighRho", e6m["Mixed_Scen1_rho-0.85"]["n_diagnostic_strikes"] + e6m["Mixed_Scen1_rho-0.95"]["n_diagnostic_strikes"], "e6")
    mint("ESixMixDiagAll", sum(e6m[f"Mixed_Scen1_rho-{x}"]["n_diagnostic_strikes"] for x in ("0.70", "0.85", "0.95")), "e6")
    mint("ESixStrikesTotal", sum(len(r["spx_bridge"]["status_per_strike"]) for r in e6["joint_rows"]), "e6")
    origin = str(e6.get("paper1_origin", ""))
    if re.search(r"[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/", origin):
        fatal("e6_joint_spx_vix.json paper1_origin still contains a local path; run scripts/sanitize_e6_origin.py")

    lab6 = {"Pure_E14_3M": r"Pure E14 ($\eta=1.9$)", "Pure_BDG6": r"Pure BDG6 ($a=1.0$)", "Mixed_Scen1": r"Mixed Scen.~1"}
    rows = []
    for idx, r in enumerate(e6["joint_rows"]):
        if idx in (3, 6):
            rows.append("\\midrule")
        fut = f"$[{dec(r['paper1_vix_fut_arb'][0], 3, 'floor')},\\,{dec(r['paper1_vix_fut_arb'][1], 3, 'ceil')}]$"
        sl = r.get("paper1_vix_smile_slope_pp")
        vsl = (f"$[{dec_signed(sl[0], 2, 'floor')},\\,{dec_signed(sl[1], 2, 'ceil')}]$" if sl is not None else "--")
        st = r["spx_bridge"]["status_per_strike"]
        atm_d = "$^{\\dagger}$" if st[4] == "diagnostic" else ""
        sk_d = "$^{\\dagger}$" if (st[3] == "diagnostic" or st[5] == "diagnostic") else ""
        base = r["model_id"].rsplit("_rho", 1)[0]
        rows.append(f"{lab6[base]} & ${r['rho']:.2f}$ & {fut} & {vsl} & {interval(r['spx_atm_iv_interval_pct'], 2)}{atm_d} "
                    f"& {interval(r['spx_atm_skew_interval'], 3)}{sk_d} & {dec(r['spx_wing_spread_put_minus_call_pp'], 2, 'nearest')} "
                    f"& {r['n_diagnostic_strikes']} \\\\")
    (OUT / "tab_e6_joint.tex").write_text(table(
        "Joint SPX--VIX audit at $H=0.10$, $T=0.25$, $\\xi_0=0.235^2$: Arb-certified VIX brackets from \\RoughCert{} v5.0.0 "
        "beside statistical \\RCBridge{} SPX envelopes.",
        "tab:e6_joint", "@{}lrrrrrrr@{}",
        "Model & $\\rho$ & VIX fut. & VIX slope & SPX ATM & SPX ATM & Wing & Diag. \\\\\n"
        " & & (pts, Arb) & (pp, Arb) & IV (\\%, stat.) & skew (stat.) & (pp) & (/8) \\\\",
        rows,
        "VIX futures and smile slopes ($k=+0.3$ minus $k=-0.3$) are the 169-bit Arb-certified results of the companion "
        "\\RoughCert{} v5.0.0 release \\citep{AhmedEtAl2026RoughCert}. SPX columns are $65{,}536$-path \\RCBridge{} (\\RCCE{}) "
        "asymptotic-envelope statistical(0.95) enclosures ($\\delta k=0.02$), or diagnostic where marked $^{\\dagger}$. "
        "Wing is the midpoint-IV difference between $K=0.85$ and $K=1.12$ (point estimate).", sep="3.0pt"), encoding="utf-8")

    # ------------------------------------------------------------------ Theorem 4 case counts (all reference envelopes)
    sets = [("E1 left-point", e1["rough_stein_stein_bridge"]["methods"]["leftpoint"]),
            ("E1 \\RCCE{}", e1["rough_stein_stein_bridge"]["methods"]["ce_i_only"]),
            ("E1 Jensen-$J$", e1["rough_stein_stein_bridge"]["methods"]["ce_adapted"]),
            ("E3 reference ($H=0.10$)", e3["ref_H010_bridge"]),
            ("E4 Regime~A reference", regA["ref_bridge"]), ("E4 Regime~B reference", regB["ref_bridge"]),
            ("E4 Regime~C reference", regC["ref_bridge"])]
    e6sum = {"i": 0, "ii": 0, "iii": 0}
    for r in e6["joint_rows"]:
        c = classify(r["spx_bridge"])
        for k in e6sum:
            e6sum[k] += c[k]
    tot = {"i": 0, "ii": 0, "iii": 0}
    rows = []
    for name, br in sets:
        c = classify(br)
        for k in tot:
            tot[k] += c[k]
        rows.append(f"{name} & {c['i']} & {c['ii']} & {c['iii']} & {sum(c.values())} \\\\")
    for k in tot:
        tot[k] += e6sum[k]
    rows.append(f"E6 (nine joint rows) & {e6sum['i']} & {e6sum['ii']} & {e6sum['iii']} & {sum(e6sum.values())} \\\\")
    rows += ["\\midrule", f"Total & {tot['i']} & {tot['ii']} & {tot['iii']} & {sum(tot.values())} \\\\"]
    mint("CaseOneCount", tot["i"], "Theorem 4 case (i) strikes over all reference envelopes")
    mint("CaseTwoCount", tot["ii"], "Theorem 4 case (ii) strikes")
    mint("CaseThreeCount", tot["iii"], "Theorem 4 case (iii) diagnostic strikes")
    mint("CaseTotalCount", sum(tot.values()), "all reference-envelope strikes")
    (SOUT / "tab_s_case_counts.tex").write_text(table(
        "Theorem~\\mref{thm:bridge_envelope} case attained by each strike of every reference \\RCBridge{} envelope.",
        "tab:s_case_counts", "@{}lrrrr@{}",
        "Envelope & Case (i) & Case (ii) & Case (iii) & Strikes \\\\\n & measured rate & assumed rate & diagnostic & \\\\",
        rows,
        "Case (i): the fitted rate is identifiable and used. Case (ii): the increments are submerged in noise and the envelope uses "
        "$\\beta=\\beta_{\\mathrm{FSW}}$, which remains valid for any true rate $\\beta\\ge\\beta_{\\min}(H)$ "
        "(Section~\\ref{sec:s_case_ii}). Case (iii): diagnostic display only."), encoding="utf-8")

    # ------------------------------------------------------------------ v2 refinement study
    joins = [("regime_A", "E4 Regime~A", regA["ref_bridge"], regA["ref_skew_interval"], regA.get("ref_bridge", {})),
             ("regime_B", "E4 Regime~B", regB["ref_bridge"], regB["ref_skew_interval"], regB.get("ref_bridge", {})),
             ("e6_pure_bdg6_rho-0.85", "E6 Pure BDG6", e6m["Pure_BDG6_rho-0.85"]["spx_bridge"],
              e6m["Pure_BDG6_rho-0.85"]["spx_atm_skew_interval"], None),
             ("e6_mixed_scen1_rho-0.85", "E6 Mixed Scen.~1", e6m["Mixed_Scen1_rho-0.85"]["spx_bridge"],
              e6m["Mixed_Scen1_rho-0.85"]["spx_atm_skew_interval"], None)]
    nv1 = nv2 = nup = nnew = 0
    n_emerging_rem = 0
    remain_rates = []
    rows = []
    for key, lab, v1b, v1skew, _ in joins:
        c = rfn["cases"][key]
        K = c["strikes"]
        if len(K) != len(v1b["status_per_strike"]):
            fatal(f"refinement strike grid differs from the reference run for {key}")
        ia = min(range(len(K)), key=lambda i: abs(K[i] - 1.0))
        d1 = [s == "diagnostic" for s in v1b["status_per_strike"]]
        d2 = [s == "diagnostic" for s in c["status_per_strike"]]
        up = sum(1 for a, bb in zip(d1, d2) if a and not bb)
        new = sum(1 for a, bb in zip(d1, d2) if bb and not a)
        nv1 += sum(d1); nv2 += sum(d2); nup += up; nnew += new
        rem = []
        for i, flag in enumerate(d2):
            if flag:
                er = c["emp_rate_per_strike"][i]
                if c["sign_flip_per_strike"][i]:
                    tag = "flip"                       # resolved sign flip
                elif er is not None:
                    remain_rates.append(er)            # slow contraction: fitted rate below beta_FSW/2
                    tag = dec(er, 2, "nearest")
                else:
                    tag = "e"                          # emerging increment: finest resolved, penultimate not
                    n_emerging_rem += 1
                rem.append(f"{k_label(K[i])} ({tag})")
        w1 = dec(v1b["iv_continuum_width_bps"][ia], 1, "ceil") + ("$^{\\dagger}$" if d1[ia] else "")
        w2 = dec(c["iv_continuum_width_bps"][ia], 1, "ceil") + ("$^{\\dagger}$" if d2[ia] else "")
        rows.append(f"{lab} & {int_tex(c['total_paths'])} & {sum(d1)} $\\to$ {sum(d2)} & {up} & {new} & {w1} $\\to$ {w2} "
                    f"& {interval(c['atm_skew_interval'], 3)} & {', '.join(rem) if rem else '--'} \\\\")
    mint("RefFlagsVOne", nv1, "v1 diagnostic flags in the four refined cases")
    mint("RefFlagsVTwo", nv2, "diagnostic flags after refinement")
    mint("RefUpgraded", nup, "flags upgraded to asymptotic-envelope")
    mint("RefNewFlags", nnew, "strikes newly flagged after refinement")
    mint("RefFinestLevel", max(rfn["levels"]), "refinement finest level")
    mint("RefEmergingRemain", n_emerging_rem, "remaining flags of the emerging-increment type")
    mint("RefSlowRemain", len(remain_rates), "remaining flags of the slow-contraction type")
    if remain_rates:
        mdec("RefRemainEmpMin", min(remain_rates), "floor", 2, "empirical rates at remaining flags")
        mdec("RefRemainEmpMax", max(remain_rates), "ceil", 2, "empirical rates at remaining flags")
    mdec("RefHalfBetaFSW", 0.5 * rfn["cases"]["regime_A"]["beta_fsw_theory"], "nearest", 2, "identifiability threshold beta_FSW/2")
    cA, cB = rfn["cases"]["regime_A"], rfn["cases"]["regime_B"]
    iK = lambda c, k: min(range(len(c["strikes"])), key=lambda i: abs(c["strikes"][i] - k))
    mdec("RefAWidthLowKVOne", regA["ref_bridge"]["iv_continuum_width_bps"][iK(cA, 0.85)], "ceil", 1, "e4 A K=0.85 width (v1)")
    mdec("RefAWidthLowKVTwo", cA["iv_continuum_width_bps"][iK(cA, 0.85)], "ceil", 1, "refined A K=0.85 width")
    mdec("RefAATMWidthVTwo", cA["iv_continuum_width_bps"][iK(cA, 1.0)], "ceil", 1, "refined A ATM width")
    mdec("RefBWidthNineFiveVOne", regB["ref_bridge"]["iv_continuum_width_bps"][iK(cB, 0.95)], "ceil", 1, "e4 B K=0.95 width (v1)")
    mdec("RefBWidthNineFiveVTwo", cB["iv_continuum_width_bps"][iK(cB, 0.95)], "ceil", 1, "refined B K=0.95")
    mdec("RefBWidthOneOFiveVOne", regB["ref_bridge"]["iv_continuum_width_bps"][iK(cB, 1.05)], "ceil", 1, "e4 B K=1.05 width (v1)")
    mdec("RefBWidthOneOFiveVTwo", cB["iv_continuum_width_bps"][iK(cB, 1.05)], "ceil", 1, "refined B K=1.05")
    for tag, key in (("A", "regime_A"), ("B", "regime_B"), ("Pure", "e6_pure_bdg6_rho-0.85"), ("Mix", "e6_mixed_scen1_rho-0.85")):
        mdec(f"Ref{tag}SkewLo", rfn["cases"][key]["atm_skew_interval"][0], "floor", 3, "refined skew lo")
        mdec(f"Ref{tag}SkewHi", rfn["cases"][key]["atm_skew_interval"][1], "ceil", 3, "refined skew hi")
    (SOUT / "tab_s_refinement.tex").write_text(table(
        f"Diagnostic-flag refinement study: the four reference envelopes with the most diagnostic flags, recomputed with finest "
        f"level $n_L=\\RefFinestLevel$ and fresh seeds.",
        "tab:s_refinement", "@{}lrrrrrr>{\\raggedright\\arraybackslash}p{0.17\\linewidth}@{}",
        "Case & Paths & Diag. & Upgr. & New & ATM width (bps) & ATM skew & Remaining flags \\\\\n"
        " & & ref.\\,$\\to$\\,refined & & & ref.\\,$\\to$\\,refined & refined & $K$ (rate) \\\\",
        rows,
        "Reference levels $(64,128,256,512)$; the refined runs add $n=1024$ at the same budget, with seeds from $20262000$. "
        "$^{\\dagger}$Diagnostic display width. Remaining flags: a number is a fitted rate below $\\beta_{\\mathrm{FSW}}/2=\\RefHalfBetaFSW$; "
        "(e) marks an emerging increment (Theorem~\\mref{thm:bridge_envelope}(iii)).",
        size="\\footnotesize", sep="3.0pt"), encoding="utf-8")

    # ------------------------------------------------------------------ E2 decomposition table (supplement)
    rows = []
    for r, a, c in zip(drows, r0, r9):
        rows.append(f"{r['H']:.2f} & {dec(a, 4, 'floor')} & {dec(r['eta_eff_star_rho0_k0_25bps'], 4, 'floor')} & {dec(c, 4, 'floor')} "
                    f"& {dec(r['factor_second_order_25bps'], 2, 'nearest')} & {dec(r['factor_leverage_25bps'], 2, 'nearest')} "
                    f"& {dec(r['factor_total_25bps'], 2, 'nearest')} \\\\")
    (SOUT / "tab_s_e2_decomposition.tex").write_text(table(
        "Decomposition of the $25\\bps$ certification boundary ($T=0.25$, $\\xi_0=0.04$, $K=1$) into a second-order (two-moment) factor "
        "and a leverage factor.",
        "tab:s_e2_decomposition", "@{}lrrrrrr@{}",
        "$H$ & $\\eta_{\\mathrm{eff}}^{*}$, $\\rho=0$ & $\\eta_{\\mathrm{eff}}^{*}$, $\\rho=0$ & $\\eta_{\\mathrm{eff}}^{*}$, $\\rho=-0.9$ "
        "& Second-order & Leverage & Total \\\\\n & two-moment & first-order & first-order & factor & factor & factor \\\\",
        rows,
        "Boundaries are the last certified points of an $\\ETwoBisectSteps$-step bisection on $[10^{-4},2.5]$ (resolution "
        "$\\ETwoBisectTol$), printed with floor rounding. Second-order factor: two-moment over first-order boundary at $\\rho=0$; "
        "leverage factor: first-order boundary at $\\rho=0$ over that at $\\rho=-0.9$; total = product (ratios of stored points)."), encoding="utf-8")

    # ------------------------------------------------------------------ exact Jensen deficit (Proposition 2) table and macros
    jd = pcj["jensen_deficit_exact"]
    best = lambda n: jd[n]["deficit_gl16"] if jd[n]["deficit_gl16"] is not None else jd[n]["deficit_gl8"]
    mdec("JenDefThirtyTwo", 100 * best("32"), "floor", 1, "exact Jensen deficit n=32 (%), GL16")
    mdec("JenDefFiveTwelve", 100 * best("512"), "floor", 1, "exact Jensen deficit n=512 (%), GL16")
    mdec("JenLBThirtyTwo", 100 * jd["32"]["lower_bound_eq"], "floor", 1, "closed-form lower bound n=32 (%)")
    mdec("JenLBFiveTwelve", 100 * jd["512"]["lower_bound_eq"], "floor", 1, "closed-form lower bound n=512 (%)")
    rho_A = -0.9
    msig("RhoEffFiveTwelve", rho_A * math.sqrt(1 - best("512")), "nearest", 3, "rho_eff at n=512 (GL16)")
    msig("RhoEffThirtyTwo", rho_A * math.sqrt(1 - best("32")), "nearest", 3, "rho_eff at n=32 (GL16)")
    rows = []
    for n in ("32", "64", "128", "256", "512"):
        r = jd[n]
        g16 = dec(100 * r["deficit_gl16"], 3, "nearest") if r["deficit_gl16"] is not None else "--"
        rows.append(f"{n} & {dec(100 * r['deficit_gl2'], 3, 'nearest')} & {dec(100 * r['deficit_gl8'], 3, 'nearest')} & {g16} "
                    f"& {dec(100 * r['lower_bound_eq'], 3, 'floor')} & ${dec_signed(rho_A * math.sqrt(1 - best(n)), 4, 'nearest')}$ \\\\")
    (SOUT / "tab_s_jensen_deficit.tex").write_text(table(
        "Exact Jensen deficit $1-\\E[I^{\\mathrm{ad}}]/\\E[I^{\\mathrm{CE}}]$ of the adapted conditional-expectation timer in Regime~A "
        "($H=0.10$, $T=0.25$, $\\eta=1.9$, $\\rho=-0.9$).",
        "tab:s_jensen_deficit", "@{}rrrrrr@{}",
        "$n$ & GL2 (\\%) & GL8 (\\%) & GL16 (\\%) & Lower bound (\\%) & $\\rho_{\\mathrm{eff}}$ \\\\",
        rows,
        "Deficits computed by Gaussian conditioning on the grid filtration with $q$-point Gauss--Legendre quadrature in each cell (GL$q$); "
        "the lower bound is the closed form of Proposition~\\mref{prop:jensen_pitfall}(i), printed with floor rounding; "
        "$\\rho_{\\mathrm{eff}}=\\rho\\sqrt{1-\\text{deficit}}$ with the most accurate deficit available."), encoding="utf-8")

    # ------------------------------------------------------------------ verification summary table (supplement)
    desc_c = {
        "C1": ("Theorem~\\mref{thm:ledger}(iii)", "Mehler identity for $\\E(\\sqrt{v}-\\sqrt{\\hat v})^2$"),
        "C2a": ("Theorem~\\mref{thm:ledger}(ii)", "ledger inequalities, random discrete laws"),
        "C2b": ("Theorem~\\mref{thm:put_lipschitz}", "Put Lipschitz bounds in $(u,S)$"),
        "C3a": ("Proposition~\\mref{prop:twomoment}", "closed-form derivatives of $\\varphi_1$ (SymPy)"),
        "C3b": ("Proposition~\\mref{prop:twomoment}", "$\\varphi_1'''>0$ on $V\\in[10^{-6},10^{3}]$"),
        "C3c": ("Proposition~\\mref{prop:twomoment}", "two-moment sandwich; sharp for two-point laws"),
        "C3d": ("Proposition~\\mref{prop:twomoment}", "$L(v)$ strictly decreasing"),
        "C3e": ("Theorem~\\mref{thm:homotopy}(a)", "first-order two-moment gap at $K=1$"),
        "C4a": ("Theorem~\\mref{thm:homotopy}(b)", "flat-$\\xi_0$ cross-covariance constant"),
        "C4b": ("Theorem~\\mref{thm:homotopy}(b)", "Gamma--Vega identity"),
        "C4c": ("Theorem~\\mref{thm:homotopy}(b)", "first-order leverage term on Heston"),
        "C5a": ("Proposition~\\mref{prop:jensen_pitfall}(i)", "small-$\\Delta t$ limit of the deficit bound"),
        "C5b": ("Proposition~\\mref{prop:jensen_pitfall}(i)", "exact deficit $\\ge$ closed-form bound"),
        "C6": ("Lemma~\\mref{lem:stopped_cv}", "stopped-timer semigroup identity"),
        "C7a": ("Theorem~\\mref{thm:bridge_envelope}(i)", "delta-method SE at SNR $3$"),
        "C7b": ("Theorem~\\mref{thm:bridge_envelope}(ii)", "E5 rates exceed $\\beta_{\\min}(H)$, $H\\ge0.10$"),
        "C8": ("Section~\\mref{sec:det}", "BLP cell-mean node and cell $L^2$ error"),
        "C9": ("Section~\\mref{sec:det}", "closed form of $R_H$ and its monotonicity"),
        "C10": ("Section~\\mref{sec:theorems}", "$\\E\\sqrt{v_s}$ closed form; optimal residual"),
    }
    desc_i = {
        "I1": ("E2 spotlights", "Arb brackets contain exact-formula brackets"),
        "I2": ("E2 boundaries", "exact width at certified points $\\le 25\\bps$"),
        "I3": ("E3 BLP hybrid", "exact $E_J$ inside Arb ball; IV mapping"),
        "I3b": ("E3 Markov", "Mehler bound and Theorem~\\mref{thm:put_lipschitz} gap"),
        "I4a": ("E1 Heston", "own CF and Lewis inversion"),
        "I4b": ("E1 Stein--Stein", "own Riccati RK4 and Lewis inversion"),
        "I5a": ("E4 Regime~A", "own \\RCCE{} vs stored level means ($|z|$)"),
        "I5b": ("E6 Pure BDG6", "own \\RCCE{} vs stored level means ($|z|$)"),
        "I6": ("E6 VIX inputs", "equality with \\RoughCert{} v5.0.0"),
        "I7": ("E4 conclusions", "BG2012 error vs refined references (bps)"),
    }
    rows = []
    for c in pcj["checks"]:
        a, d = desc_c[c["id"]]
        rows.append(f"{c['id']} & {a} & {d} & ${sci(c['value'], 2, 'nearest')}$ & ${threshold_tex(c['tolerance'])}$ & "
                    f"{'pass' if c['pass'] else 'FAIL'} \\\\")
    rows.append("\\midrule")
    for c in icj["checks"]:
        a, d = desc_i[c["id"]]
        rows.append(f"{c['id']} & {a} & {d} & ${sci(c['value'], 2, 'nearest')}$ & ${threshold_tex(c['tolerance'])}$ & "
                    f"{'pass' if c['pass'] else 'FAIL'} \\\\")
    mint("ProofChecksN", pcj["n_checks"], "v2_proof_checks"); mint("IndepChecksN", icj["n_checks"], "v2_independent_check")
    i4a = next(c for c in icj["checks"] if c["id"] == "I4a"); i4b = next(c for c in icj["checks"] if c["id"] == "I4b")
    msci("IndepHestonMaxDiff", i4a["value"], "ceil", 1, "I4a"); msci("IndepSSMaxDiff", i4b["value"], "ceil", 1, "I4b")
    mdec("IndepZMaxA", next(c for c in icj["checks"] if c["id"] == "I5a")["value"], "ceil", 2, "I5a")
    mdec("IndepZMaxB", next(c for c in icj["checks"] if c["id"] == "I5b")["value"], "ceil", 2, "I5b")
    mdec("IndepETwoMaxExactWidth", next(c for c in icj["checks"] if c["id"] == "I2")["value"], "ceil", 2, "I2")
    (SOUT / "tab_s_checks.tex").write_text(table(
        f"Verification summary: \\ProofChecksN{{}} first-principles checks of the stated results (C) and \\IndepChecksN{{}} "
        f"code-disjoint recomputations of archived numbers (I).",
        "tab:s_checks", "@{}lllrrl@{}",
        "ID & Anchor & Quantity checked & Metric & Tolerance & Result \\\\",
        rows,
        "C: first-principles checks (mpmath at 40 digits, SymPy); I: code-disjoint recomputations; both scripts are listed in "
        "Table~\\ref{tab:s_manifest} and import no production code. Metrics are maximum deviations or violations unless stated "
        "(a negative violation is a margin); tolerances are the pass thresholds.",
        size="\\scriptsize", sep="3.0pt"), encoding="utf-8")

    # ------------------------------------------------------------------ write macros and rules
    lines = ["% Auto-generated by scripts/build_numbers_and_tables.py from results/*.json -- DO NOT EDIT BY HAND",
             "% Rounding rules per macro: manuscript/tables/spx_numbers_rules.json"]
    lines += [f"\\newcommand{{\\{k}}}{{{REG[k]['printed']}}}" for k in sorted(REG)]
    (OUT / "spx_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "spx_numbers_rules.json").write_text(json.dumps(REG, indent=1, sort_keys=True), encoding="utf-8")
    print(f"Generated {len(REG)} macros, 6 main tables and 5 supplementary tables.")


if __name__ == "__main__":
    main()
