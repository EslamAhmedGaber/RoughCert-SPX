# RoughCert-SPX: code and data

Code, archived results and manuscript sources for *RoughCert-SPX: Certified-Bias SPX Option Pricing and Kernel Audits
under Correlated Rough Volatility* (Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman, Sroor M. Elnady).
The numerical exhibits are generated from the JSON files in
`results/` by the scripts in `scripts/`.

## Layout

| Folder | Content |
|---|---|
| `src/` | Engines: Arb Black-Scholes and implied volatility (`bs_arb.py`), Volterra kernels and Arb kernel norms (`volterra_arb.py`), deterministic SPX brackets (`deterministic_spx.py`), exact-bridge RQMC envelopes (`bridge_rqmc.py`), exact laws (`exact_laws.py`) and the published competitors (`competitors.py`) |
| `scripts/` | Experiment drivers, verification scripts, number/table/figure generators and the build gate |
| `results/` | Archived results of every experiment and verification run (JSON) |
| `manuscript/` | Modular LaTeX sources: `main.tex` (master), `one_column.tex`, `two_columns.tex`, `sections/`, `tables/` (generated), `figures/` (generated), `supplement/` |

## Scripts and the artefacts they produce

| Script | Archive | Use |
|---|---|---|
| `scripts/run_unit_verifications.py` | `assertion log` | Arb Black-Scholes and IV, kernel norms, exact laws, stopped-timer control, BG2012 orders, domain guards |
| `scripts/run_cv_ablation.py` | `cv_ablation.json` | Control-bank variance reduction (Section 3.3) |
| `scripts/run_e1_exact_laws.py` | `e1_exact_laws.json` | Experiment E1 |
| `scripts/run_e2_det_phase_map.py` | `e2_det_phase_map.json` | Experiment E2 |
| `scripts/run_e3_kernel_audit.py` | `e3_kernel_audit.json` | Experiment E3 |
| `scripts/run_e4_benchmarks.py` | `e4_benchmarks.json` | Experiment E4 |
| `scripts/run_e5_weak_rates.py` | `e5_weak_rates.json` | Experiment E5 |
| `scripts/run_e6_joint_spx_vix.py` | `e6_joint_spx_vix.json` | Experiment E6 (VIX inputs from RoughCert v5.0.0, pinned by SHA-256) |
| `scripts/run_v2_diag_refinement.py` | `v2_diag_refinement.json` | Supplementary Table S1 |
| `scripts/run_v2_e2_decomposition.py` | `v2_e2_decomposition.json` | Supplementary Table S2 |
| `scripts/v2_proof_checks.py` | `v2_proof_checks.json` | C-checks, Supplementary Tables S4-S5 |
| `scripts/v2_independent_check.py` | `v2_independent_check.json` | I-checks, Supplementary Table S5 |
| `scripts/build_numbers_and_tables.py` | `spx_numbers.tex, rules, tables` | All macros and tables (directed rounding) |
| `scripts/build_figures.py` | `figures/fig_*.pdf` | Figures of the main article |
| `scripts/build_verify.py` | `PDFs, build_qa.json` | Builds both layouts and this supplement; runs the gates |
| `scripts/v2_number_audit.py` | (report only) | Re-derives every macro from the archives with exact arithmetic and checks the rounding rules |

## Reproduction

1. `python -m pip install -r requirements.txt` (tested versions; Python 3.12).
2. `python -B scripts/run_unit_verifications.py` runs seven test groups (about three minutes on one thread).
3. `python -B scripts/build_numbers_and_tables.py` regenerates `manuscript/tables/` from `results/`; the output is
   byte-identical to the shipped files. `python -B scripts/v2_number_audit.py` re-derives every macro independently.
4. `python -B scripts/build_figures.py` regenerates `manuscript/figures/`.
5. `python -B scripts/v2_proof_checks.py` (19 first-principles checks, about two minutes) and
   `python -B scripts/v2_independent_check.py` (10 code-disjoint recomputations) rewrite the verification archives.
   Check I6 compares all nine futures pairs and the two complete smile arrays with the bundled, SHA-256-pinned companion v5.0.0 files.
6. The experiment drivers `scripts/run_e1_exact_laws.py` ... `scripts/run_e6_joint_spx_vix.py`,
   `scripts/run_cv_ablation.py`, `scripts/run_v2_diag_refinement.py` and `scripts/run_v2_e2_decomposition.py` rerun the
   experiments from fixed seeds (single thread; several hours in total). Their outputs replace the archives in `results/`.
7. Manuscript: in `manuscript/`, run `pdflatex one_column`, `bibtex one_column` and `pdflatex one_column` twice (likewise
   `two_columns`); then, in `manuscript/supplement/`, the same sequence for `supplement` (it reads `../one_column.aux`
   for cross-references). `python -B scripts/build_verify.py` performs all builds and the submission gates; it needs
   TeX Live and the scientific-writing-studio QA scripts (environment variables `TEXBIN` and `SWS_SCRIPTS`).

## Conventions

* Arb ball arithmetic runs at 52 decimal digits, which python-flint maps to 176 bits. Lower endpoints are rounded down
  and upper endpoints up when converted to floating point.
* Printed numbers follow `manuscript/tables/spx_numbers_rules.json`: lower bounds, certified lows and "at least"
  ratios are rounded down; upper ends, widths, errors and thresholds up; point estimates to nearest; ranges round the
  minimum down and the maximum up.
* Epistemic labels: *Arb-certified* (machine-verified interval), *asymptotic-envelope statistical(0.95)* (exact-bridge
  envelope under Case (i) or (ii) of Theorem 4), *statistical(0.95)* (sampling interval only) and *diagnostic*
  (no guarantee).
* All runs use one thread (`OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`) and fixed seeds recorded
  in the result files.

Public software and data release [v3.0.0](https://github.com/EslamAhmedGaber/RoughCert-SPX/releases/tag/v3.0.0). Companion working paper and software [v5.0.0](https://github.com/EslamAhmedGaber/RoughCert/releases/tag/v5.0.0).
