"""Build and verify RoughCert-SPX: both Elsevier layouts, the supplement, and every text gate.

Steps
  1. Regenerate macros and tables from results/*.json (scripts/build_numbers_and_tables.py) and run the
     independent number gate (scripts/v2_number_audit.py).
  2. one_column.tex and two_columns.tex: pdflatex, bibtex, pdflatex, pdflatex (plus one stabilising pass if
     labels changed), then a fourth pass whose .aux must be byte-identical to the previous one.
  3. supplement/supplement.tex: three pdflatex passes after the one-column build (xr-hyper reads ../one_column.aux).
  4. Every log must show zero errors, warnings, undefined references or citations, missing characters, and
     overfull or underfull boxes.
  5. scientific-writing-studio gates: check_norms (flattened, macros substituted), check_tex_hygiene,
     check_technical --team-citation, scan_leaks (only the approved AI sentence allowed).
  6. results/build_qa.json records every step with repository-relative paths only.

Environment: SWS_SCRIPTS (folder of the scientific-writing-studio QA scripts; required), TEXBIN (folder of
pdflatex and bibtex; default: found on PATH), TEXMFVAR (default <repo>/.texmf-var).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manuscript"
SUP = M / "supplement"
A = ROOT / "audit" / "build"
FUND = ("This work was supported by the Science, Technology & Innovation Funding Authority (STDF), Egypt, "
        "under the Basic Sciences Grant, Project ID 50925.")


def declaration_allow_pattern() -> str:
    """scan_leaks --allow pattern for the manuscript's approved generative-AI declaration, read from the source."""
    text = (M / "sections" / "09_declarations.tex").read_text(encoding="utf-8")
    import hashlib
    sentence = text.strip().splitlines()[-1]
    if hashlib.sha256(sentence.encode("utf-8")).hexdigest() != "8cfa54d605f3f1405f52792fa4ad92809aa4e12d75b11504db7e26ae1b4f26b3":
        raise ValueError("The writing declaration differs from the approved wording")
    return "*09_declarations.tex:" + re.escape(sentence.split(" used ", 1)[1].split(" of the ", 1)[0])



def sws_dir() -> Path:
    c = os.environ.get("SWS_SCRIPTS")
    if c and (Path(c) / "check_norms.py").is_file():
        return Path(c)
    raise FileNotFoundError("scientific-writing-studio QA scripts not found; set SWS_SCRIPTS to their folder")


def tex_env() -> dict:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(ROOT / ".mplconfig")}
    texbin = os.environ.get("TEXBIN")
    if not texbin and not shutil.which("pdflatex"):
        raise FileNotFoundError("pdflatex not found; set TEXBIN to the TeX Live binary folder")
    if texbin:
        env["PATH"] = texbin + os.pathsep + env.get("PATH", "")
    var = os.environ.get("TEXMFVAR") or str(ROOT / ".texmf-var")
    env.setdefault("TEXMFVAR", var)
    env.setdefault("TEXMFCACHE", var)
    return env


ENV = tex_env()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return p.name


def run(name: str, cmd: list[str], cwd: Path) -> dict:
    A.mkdir(parents=True, exist_ok=True)
    log = A / f"{name}.txt"
    exe = shutil.which(str(cmd[0]), path=ENV["PATH"]) if not os.path.isabs(str(cmd[0])) else None
    cmd_abs = [exe or cmd[0], *cmd[1:]]   # Windows resolves executables from the parent PATH otherwise
    with log.open("wb") as f:
        r = subprocess.run(cmd_abs, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    shown = [Path(c).name if (os.path.isabs(str(c))) else str(c) for c in cmd]
    print(f"{name:28s} EXIT={r.returncode}", flush=True)
    return {"name": name, "command": shown, "cwd": rel(cwd), "exit": r.returncode, "log": rel(log)}


def log_issues(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    pats = {
        "errors": r"^! ",
        "latex_warnings": r"LaTeX Warning:",
        "package_warnings": r"Package \S+ Warning:",
        "class_warnings": r"Class \S+ Warning:",
        "pdftex_warnings": r"pdfTeX warning",
        "undefined": r"(Reference|Citation) `[^']*' .*undefined|There were undefined",
        "missing_characters": r"Missing character",
        "overfull": r"Overfull \\[hv]box",
        "underfull": r"Underfull \\[hv]box",
    }
    out = {k: len(re.findall(p, txt, flags=re.M)) for k, p in pats.items()}
    out["examples"] = [ln.strip()[:160] for ln in txt.splitlines()
                       if re.search("|".join(pats.values()), ln)][:12]
    return out


def build_doc(doc: str, folder: Path, with_bib: bool, results: list) -> dict:
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out", ".toc"):
        (folder / f"{doc}{suffix}").unlink(missing_ok=True)
    tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"{doc}.tex"]
    seq = [tex, ["bibtex", doc], tex, tex] if with_bib else [tex, tex, tex]
    for i, cmd in enumerate(seq, 1):
        step = run(f"{doc}_{i}", cmd, folder)
        results.append(step)
        if step["exit"]:
            return {"doc": doc, "ok": False}
    logtxt = (folder / f"{doc}.log").read_text(encoding="utf-8", errors="replace")
    if "Label(s) may have changed" in logtxt or "Rerun to get" in logtxt:
        results.append(run(f"{doc}_stabilize", tex, folder))
    aux_before = (folder / f"{doc}.aux").read_bytes()
    results.append(run(f"{doc}_fourth_pass", tex, folder))
    aux_after = (folder / f"{doc}.aux").read_bytes()
    import fitz  # PyMuPDF
    with fitz.open(folder / f"{doc}.pdf") as pdf:
        pages = len(pdf)
    issues = log_issues(folder / f"{doc}.log")
    info = {"doc": doc, "ok": True, "pages": pages, "aux_invariant_4th_pass": aux_before == aux_after, **issues}
    print(f"   {doc}: {pages} pp, aux invariant={aux_before == aux_after}, "
          + ", ".join(f"{k}={v}" for k, v in issues.items() if k != "examples"), flush=True)
    return info


def expanded_copy(main_tex: Path, out: Path) -> Path:
    """Flatten main_tex (inputs inlined, comments removed) and substitute every generated macro by its value."""
    sys.path.insert(0, str(sws_dir()))
    from flatten_tex import flatten  # noqa: E402
    text = flatten(str(main_tex))
    defs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$",
                           (M / "tables" / "spx_numbers.tex").read_text(encoding="utf-8"), flags=re.M))
    for k in sorted(defs, key=len, reverse=True):
        text = re.sub(r"\\" + k + r"(?![A-Za-z])(\{\})?", lambda _m, v=defs[k]: v, text)
    out.write_text(text, encoding="utf-8")
    return out


def stage_sources(dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    for p in M.rglob("*"):
        if p.is_file() and p.suffix in (".tex", ".bib", ".pdf", ".json") and p.name not in (
                "one_column.pdf", "two_columns.pdf", "main.pdf", "supplement.pdf"):
            q = dst / p.relative_to(M)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    return dst


def main() -> int:
    results: list[dict] = []
    py = [sys.executable, "-B"]
    results.append(run("build_numbers_and_tables", py + [str(ROOT / "scripts" / "build_numbers_and_tables.py")], ROOT))
    results.append(run("v2_number_audit", py + [str(ROOT / "scripts" / "v2_number_audit.py")], ROOT))
    docs = [build_doc("one_column", M, True, results), build_doc("two_columns", M, True, results),
            build_doc("supplement", SUP, False, results)]
    qa = sws_dir()
    flat = expanded_copy(M / "one_column.tex", A / "one_column_expanded.tex")
    flat_s = expanded_copy(SUP / "supplement.tex", A / "supplement_expanded.tex")
    stage = stage_sources(A / "scan_stage")
    allow = declaration_allow_pattern()   # the approved generative-AI sentence is the only permitted tool-name match
    gates = [
        ("check_norms_main", [str(flat)], M),
        ("check_norms_supplement", [str(flat_s)], SUP),
        ("check_tex_hygiene_main", ["main.tex"], M),
        ("check_tex_hygiene_supplement", ["supplement.tex"], SUP),
        ("check_technical", ["main.tex", "--pdf", "one_column.pdf", "--bbl", "one_column.bbl", "--bib", "references.bib",
                             "--order", "refs-then-appendix", "--declarations", "before-refs", "--funding", FUND,
                             "--team-citation"], M),
        ("scan_leaks_sources", [str(stage), "--allow", allow], ROOT),
    ]
    for name, args, wd in gates:
        results.append(run(name, py + [str(qa / (name.split("_main")[0].split("_supplement")[0].split("_sources")[0] + ".py")), *args], wd))
    ok_docs = all(d.get("ok") and d["aux_invariant_4th_pass"] and all(
        d[k] == 0 for k in ("errors", "latex_warnings", "package_warnings", "class_warnings", "pdftex_warnings",
                            "undefined", "missing_characters", "overfull", "underfull")) for d in docs)
    summary = {"steps": results, "documents": docs, "all_exit_zero": all(r["exit"] == 0 for r in results),
               "documents_clean": ok_docs}
    (ROOT / "results" / "build_qa.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL EXIT 0: {summary['all_exit_zero']}; documents clean: {ok_docs}")
    return 0 if (summary["all_exit_zero"] and ok_docs) else 1


if __name__ == "__main__":
    sys.exit(main())
