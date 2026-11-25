"""Independent gate for the generated numbers of RoughCert-SPX (v2).

1. Re-derives every printed macro from its archived raw value with exact rational arithmetic
   (fractions.Fraction, independent of the Decimal code in build_numbers_and_tables.py) under the
   recorded rule (floor / ceil / nearest) and compares with manuscript/tables/spx_numbers.tex.
2. Requires every generated macro to be used in the manuscript or the supplement.
3. Checks that every printed range or interval (macro pairs ...Min/...Max and ...Lo/...Hi) contains
   the raw values it summarises.
4. Scans the flattened prose for phrases removed in the v2 round (overclaims, wrong precision, wrong
   scrambling name, doubled appendix labels) and for hand-typed copies of computed v1 numbers.
5. Writes audit/v2_change_log.json: every v1 macro with its v1 and v2 printed value, the rule and the
   reason, every new macro, and the curated list of prose-level number changes.

Exit status 0 only if all checks pass. Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript"
V1_NUMBERS = ROOT.parent / "13_P2_ROUGHCERT_SPX" / "manuscript" / "tables" / "spx_numbers.tex"   # read only

FORBIDDEN = [
    "173-bit", "173 bits", "Owen-scrambled", "Owen-Sobol", "100\\%", "unconditional", "statistically proven",
    "provably", "Direct confirmation", "confirming Theorem", "verified IV", "verified implied-volatility",
    "nearly halves", "halves coarse-grid", "in milliseconds", "in under one second", "\\hyperref[app:",
    "Appendix~\\ref", "Appendix \\ref", "Certified joint", "certified joint", "far below time-stepping",
]
REMOVED = {
    "EFourAJenTwoFiveSixATMLo": "point-estimate distance (3.9 bps) mixed sampling error with bias; replaced by the CI-based separation \\EFourAJenTwoFiveSixATMCIDist",
    "EOneRSSCeContained": "replaced by the prose statement that all seven reference prices are contained (all three envelopes)",
}
PARAMS = {  # model and grid parameters that coincide with computed values (strike grids, maturities, correlations)
    "0.85", "0.90", "0.95", "1.00", "1.05", "1.10", "1.15", "0.10", "0.25", "0.50", "0.70", "0.30", "0.40", "0.45", "0.20",
}
V1_RULES = {  # rounding used by the v1 generator for the macros whose printed value changed
    "EOneRSSCeGainLow": "nearest (v1 fmt_pt)", "EOneRSSMart": "nearest (v1 fmt_sci_tex)",
    "EOneSSRiccati": "nearest (v1 fmt_sci_tex)", "ETwoSSevenRatio": "nearest (v1 fmt_pt)",
}
PROSE = [  # curated prose-level number changes (v1 text -> v2 text); all v2 numbers are generated macros
    ("Sec. 3.2", "deficit exceeds 25% at n=32, above 15% at n=512 (hand-typed)", "\\JenDefThirtyTwo = 25.1%, \\JenDefFiveTwelve = 15.3% (exact deficit, GL16, floor)"),
    ("Sec. 3.2", "rho_eff from -0.900 to -0.828 (hand-typed)", "\\RhoEffThirtyTwo = -0.779 (n=32), \\RhoEffFiveTwelve = -0.828 (n=512)"),
    ("Sec. 4.1", "residual truncation uncertainty <~ 0.5 bps (not derived)", "\\EOneRSSRefDiffIV = 0.3 bps (first-order IV equivalent of the M=128->192 refinement)"),
    ("Sec. 4.1", "Channel-I projection nearly halves coarse-grid error (ratio 1.79, nearest)", "reduction by a factor of at least \\EOneRSSCeGainLow = 1.78 (floor)"),
    ("Sec. 4.1", "Jensen-J n=256 bias nearly twice left-point's", "\\EOneRSSJenOverLPFine = 1.76 times (floor)"),
    ("Sec. 4.2", "two-moment boundary an order of magnitude wider than first order", "\\ETwoSecondOrderMin--\\ETwoSecondOrderMax = 7.6--11.8 times"),
    ("Abstract/Sec. 4.2", "43x--68x equity-leverage collapse (attributed to leverage)", "43--68 = leverage factor \\ETwoLeverageMin--\\ETwoLeverageMax (5.75--5.77) x second-order factor (7.6--11.8)"),
    ("Sec. 4.2", "Bridge tightens S7 by 288x (nearest)", "narrower by a factor of at least \\ETwoSSevenRatio = 287 (floor)"),
    ("Sec. 5.2", "statistically proven ATM bias of at least 3.9 bps (point estimate)", "95% CI lies \\EFourAJenTwoFiveSixATMCIDist = 2.3 bps above the reference envelope"),
    ("Sec. 5.2", "variance ratio 1.000; 0.805 at n=64 (hand-typed)", "\\EFourARDVmVarRatio = 1.000; \\EFourARDBsVarRatioSixtyFour = 0.805"),
    ("Sec. 5.1", "M>=12 continuous-time kernel gap already small", "certified M=16 gap ~ \\EThreeMkTenMSixteenGapIV = 84 bps vs empirical <= 4.6 bps (factor >= 18)"),
    ("Sec. 6", "VIX slope lifted by over five percentage points", "certified separation at least \\ESixVixLiftLo = 5.03 pp (floor of mixed lo - pure hi)"),
    ("Sec. 7", "kappa=2 drives kernel bias far below time-stepping error", "removed: certified kappa=2 IV gap is 55.6 bps (conditional)"),
    ("Sec. 7", "RC-CE preserves the full FSW rate and halves coarse-grid error", "measured rates (E5); coarse-grid factor 1.78"),
    ("Theorem 4", "skew interval label statistical(0.95)", "joint coverage >= 1 - 2 alpha = 0.90 (Bonferroni over two strikes)"),
    ("All", "173-bit Arb", "\\ArbBits = 176 bits (ARB_DPS = 52 in src/bs_arb.py)"),
    ("Supplement Table S1", "reference runs: levels (32,64,128,256) (first draft of the v2 note)", "levels (64,128,256,512) as archived"),
    ("Sec. 3.2", "attenuation of order n^{-2H} expected to bias prices at the same order; E1, E4, E5 test it (outcome not reported)",
     "E5 ablation rates at K = 1.06 below those of RC-CE at every H; every resolved ablation rate exceeds 2H (heuristic order not observed)"),
    ("Sec. 5.3", "ablation rates lower throughout the rough regime (0.778 vs 0.782 at H = 0.15)",
     "negligible gap at H = 0.15, where the ATM (0.589 vs 0.932) and strip-mean (0.663 vs 0.863) rates remain lower"),
]


def macros_of(path: Path) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", path.read_text(encoding="utf-8"), flags=re.M))


def frac_round(x: Fraction, nd: int, rule: str) -> Fraction:
    s = Fraction(10) ** nd
    y = x * s
    if rule == "floor":
        q = y.numerator // y.denominator
    elif rule == "ceil":
        q = -((-y.numerator) // y.denominator)
    else:  # nearest, ties to even
        f = y.numerator // y.denominator
        r = y - f
        q = f + (1 if (r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2 == 1)) else 0)
    return Fraction(q) / s


def fmt_fixed(v: Fraction, nd: int) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    s = Fraction(10) ** nd
    n = v * s
    assert n.denominator == 1
    n = n.numerator
    ip, fp = divmod(n, 10 ** nd) if nd > 0 else (n, 0)
    out = f"{ip}" + (f".{fp:0{nd}d}" if nd > 0 else "")
    return ("" if out.strip("0.") == "" else sign) + out


def rederive(rec: dict) -> str | None:
    raw, rule, nd, printed = rec["raw"], rec["rule"], rec["digits"], rec["printed"]
    if rule == "exact" or raw is None:
        return None
    if float(raw) == 0.0 and printed == "0":
        return "0"                      # sci() prints an exact zero as 0
    x = Fraction(float(raw))
    if "\\times 10^" in printed:
        if x == 0:
            return "0"
        e = 0
        a = abs(x)
        while a >= 10:
            a /= 10; e += 1
        while a < 1:
            a *= 10; e -= 1
        m = frac_round(x / Fraction(10) ** e, nd, rule)
        if abs(m) >= 10:
            e += 1
            m = frac_round(x / Fraction(10) ** e, nd, rule)
        return f"{fmt_fixed(m, nd)}\\times 10^{{{e}}}"
    v = frac_round(x, nd, rule)
    s = fmt_fixed(v, nd)
    if printed.startswith("+"):
        return "+" + s if v > 0 else s
    return s


def flatten(path: Path, seen=None) -> str:
    seen = seen or set()
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%.*", "", text)

    def repl(m):
        name = m.group(1)
        cand = (path.parent / name)
        cand = cand if cand.suffix == ".tex" else cand.with_suffix(".tex")
        if not cand.exists() or cand in seen or cand.name == "spx_numbers.tex":
            return ""
        seen.add(cand)
        return flatten(cand, seen)
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def main() -> int:
    fails: list[str] = []
    rules = json.loads((MAN / "tables" / "spx_numbers_rules.json").read_text(encoding="utf-8"))
    printed = macros_of(MAN / "tables" / "spx_numbers.tex")
    if set(rules) != set(printed):
        fails.append(f"rules/macros mismatch: {sorted(set(rules) ^ set(printed))}")
    n_checked = 0
    for k, rec in rules.items():
        want = rederive(rec)
        if want is None:
            continue
        n_checked += 1
        if want != printed.get(k):
            fails.append(f"re-derivation mismatch {k}: printed {printed.get(k)} vs independent {want}")
    # usage
    corpus = flatten(MAN / "main.tex") + flatten(MAN / "supplement" / "supplement.tex")
    unused = [k for k in printed if not re.search(r"\\" + k + r"(?![A-Za-z])", corpus)]
    if unused:
        fails.append(f"unused macros: {unused}")
    # range containment
    n_pairs = 0
    for k, rec in rules.items():
        for lo_s, hi_s in (("Min", "Max"), ("Lo", "Hi")):
            if k.endswith(lo_s):
                k2 = k[: -len(lo_s)] + hi_s
                if k2 in rules and rec["raw"] is not None and rules[k2]["raw"] is not None:
                    lo_p = printed[k].lstrip("+"); hi_p = printed[k2].lstrip("+")
                    if "times" in lo_p or "times" in hi_p:
                        continue
                    n_pairs += 1
                    if Fraction(lo_p) > Fraction(float(rec["raw"])) or Fraction(hi_p) < Fraction(float(rules[k2]["raw"])):
                        fails.append(f"range {k}/{k2} printed [{lo_p},{hi_p}] excludes raw [{rec['raw']},{rules[k2]['raw']}]")
    # forbidden phrases in the prose (generated tables included)
    scan = corpus.replace("\\texttt{Owen-Sobol RQMC}", "")   # App. B names the archived label on purpose
    hits = [p for p in FORBIDDEN if p in scan]
    if hits:
        fails.append(f"forbidden phrases present: {hits}")
    # hand-typed copies of computed v1 numbers
    v1 = macros_of(V1_NUMBERS) if V1_NUMBERS.exists() else {}
    prose = "\n".join(re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8")) for p in sorted((MAN / "sections").glob("*.tex")))
    prose += re.sub(r"(?<!\\)%.*", "", (MAN / "supplement" / "supplement.tex").read_text(encoding="utf-8"))
    typed = []
    for k, val in v1.items():
        if val in PARAMS:
            continue
        if re.fullmatch(r"-?\d+\.\d{2,}", val) and len(val) >= 4:
            if re.search(r"(?<![\d.])" + re.escape(val) + r"(?![\d])", prose):
                typed.append((k, val))
    if typed:
        fails.append(f"hand-typed computed numbers in prose: {typed}")
    # change log
    log = {"generator": "scripts/build_numbers_and_tables.py", "rules_file": "manuscript/tables/spx_numbers_rules.json",
           "macros_changed": [], "macros_unchanged": 0, "macros_new": {}, "prose_changes": [
               {"location": a, "v1": b, "v2": c} for a, b, c in PROSE]}
    for k in sorted(v1):
        if k not in printed:
            log["macros_changed"].append({"macro": k, "v1": v1[k], "v2": None, "reason": "removed: " + REMOVED.get(k, "unused")})
        elif v1[k] != printed[k]:
            log["macros_changed"].append({"macro": k, "v1": v1[k], "v2": printed[k], "raw": rules[k]["raw"],
                                          "v2_rule": rules[k]["rule"], "v1_rule": V1_RULES.get(k, "see v1 generator"),
                                          "reason": "directed rounding applied to the exact binary value"})
        else:
            log["macros_unchanged"] += 1
    log["macros_new"] = {k: printed[k] for k in sorted(set(printed) - set(v1))}
    (ROOT / "audit" / "v2_change_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"re-derived {n_checked} macros, {n_pairs} ranges, {len(printed)} macros used, "
          f"{len(log['macros_changed'])} v1 values changed, {len(log['macros_new'])} new macros")
    for f in fails:
        print("FAIL:", f)
    print("EXIT", 1 if fails else 0)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
