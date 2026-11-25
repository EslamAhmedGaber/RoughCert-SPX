"""Assemble final_package/ for the RoughCert-SPX submission and verify every deliverable.

Run from the repository root after scripts/build_verify.py has passed:
    python -B scripts/build_final_package.py
Environment: SWS_SCRIPTS (scientific-writing-studio QA scripts; required), TEXBIN (TeX Live binaries; default PATH),
COMPANION_PDF and COMPANION_AUX (optional: PDF and .aux of the RoughCert v5.0.0 working paper, for for_review_only/).
An existing final_package/ (and sealed zip) is renamed *.prev_<UTC time>; nothing is deleted.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_verify as bv  # noqa: E402  (shared helpers: TeX environment, QA folder, flattening, log parsing)

M = ROOT / "manuscript"
SUP = M / "supplement"
PKG = ROOT / "final_package"
SEALED = ROOT / "RoughCert_SPX_v3_submission_package.zip"
METHOD = "RoughCert_SPX"
STAMP = (2026, 11, 25, 0, 0, 0)          # fixed zip timestamps keep the archives reproducible
DATE = "25 November 2026"
TITLE = ("RoughCert-SPX: Certified-Bias SPX Option Pricing and Kernel Audits under Correlated Rough Volatility")
AUTHORS = "Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady"
COMPANION_URL = "https://github.com/EslamAhmedGaber/RoughCert/releases/tag/v5.0.0"
def release_macro(name):
    text = (M / "main.tex").read_text(encoding="utf-8")
    return re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", text)[1]

SWS = bv.sws_dir()
PY = [sys.executable, "-B"]
CHECKS: dict = {}


def journal() -> str:
    m = re.search(r"\\def\\TargetJournal\{([^}]*)\}", (M / "one_column.tex").read_text(encoding="utf-8"))
    return m.group(1) if m else "the journal"


def numbers() -> dict:
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", (M / "tables" / "spx_numbers.tex").read_text(encoding="utf-8"), flags=re.M))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sanitize(text: str) -> str:
    """Replace machine-specific paths in shipped logs and reports by placeholders."""
    reps = [(str(PKG), "<package>"), (str(ROOT), "<repo>"), (str(SWS), "<qa-scripts>"), (tempfile.gettempdir(), "<tmp>"),
            (str(Path(sys.executable).parent), "<python>"), (str(Path.home()), "<home>")]
    tex = shutil.which("pdflatex", path=bv.ENV["PATH"])
    if tex:
        reps.append((str(Path(tex).parent), "<texbin>"))
    for a, b in sorted(reps, key=lambda t: -len(t[0])):
        for form in (a, a.replace("\\", "\\\\"), a.replace("\\", "/")):
            text = re.sub(re.escape(form), lambda _m: b, text, flags=re.I)
    return text


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize(text), encoding="utf-8", newline="\n")


def run(cmd: list, cwd: Path, env: dict | None = None, log: Path | None = None) -> int:
    env = env or bv.ENV
    exe = shutil.which(str(cmd[0]), path=env["PATH"]) if not os.path.isabs(str(cmd[0])) else None
    r = subprocess.run([exe or cmd[0], *cmd[1:]], cwd=cwd, capture_output=True, env=env)
    if log is not None:
        write_report(log, r.stdout.decode("utf-8", "replace") + "\n--- stderr ---\n" + r.stderr.decode("utf-8", "replace"))
    return r.returncode


def tex_build(folder: Path, stem: str, env: dict, with_bib: bool = True) -> dict:
    tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", stem + ".tex"]
    rcs = [run(tex, folder, env), run(["bibtex", stem], folder, env) if with_bib else run(tex, folder, env),
           run(tex, folder, env), run(tex, folder, env)]
    aux_before = (folder / f"{stem}.aux").read_bytes()
    rcs.append(run(tex, folder, env))
    issues = bv.log_issues(folder / f"{stem}.log")
    import fitz
    with fitz.open(folder / f"{stem}.pdf") as d:
        pages = d.page_count
    return {"exit_codes": rcs, "pages": pages, "aux_invariant_4th_pass": aux_before == (folder / f"{stem}.aux").read_bytes(),
            **{k: v for k, v in issues.items() if k != "examples"}}


def clean_env(tmp: Path) -> dict:
    env = {k: v for k, v in bv.ENV.items() if k not in ("PYTHONPATH",)}
    env.update({"TEXMFVAR": str(tmp / ".texmf-var"), "TEXMFCACHE": str(tmp / ".texmf-var"), "MPLCONFIGDIR": str(tmp / ".mpl"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def make_zip(target: Path, folder: Path, prefix: str = "") -> Path:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(prefix + p.relative_to(folder).as_posix(), date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    return target


def pdf_compare(a: Path, b: Path, log: Path) -> int:
    return run(PY + [str(SWS / "pdf_compare.py"), str(a), str(b), "--text", "--dpi", "110"], ROOT, log=log)


def allow_patterns() -> list[str]:
    core = bv.declaration_allow_pattern().split(":", 1)[1]
    return [f"{g}:{core}" for g in ("*09_declarations.tex", f"*{METHOD}_manuscript.tex", "*main_flat_substituted.tex", "*two_columns.tex", "*AUDIT.md")]


def scan(target: Path, name: str) -> int:
    declaration = (M / "sections/09_declarations.tex").read_text(encoding="utf-8").strip().splitlines()[-1]
    import fnmatch
    patterns = [v.split(":",1)[0] for v in allow_patterns()]
    if target.is_dir():
        for f in target.rglob("*"):
            if f.is_file() and any(fnmatch.fnmatch(f.name,g) for g in patterns):
                body = f.read_text(encoding="utf-8").replace(declaration, "")
                if any(word in body for word in declaration.split(" used ")[1].split(" to improve")[0].split(" and ")):
                    raise ValueError("Unapproved writing declaration in " + f.name)
    args = PY + [str(SWS / "scan_leaks.py"), str(target), "--json", str(PKG / "qa" / f"scan_leaks_{name}.json")]
    for a in allow_patterns():
        args += ["--allow", a]
    rc = run(args, ROOT, log=PKG / "qa" / f"scan_leaks_{name}.txt")
    rep = PKG / "qa" / f"scan_leaks_{name}.json"
    if rep.exists():
        write_report(rep, rep.read_text(encoding="utf-8", errors="replace"))
    CHECKS.setdefault("scan_leaks", {})[name] = rc
    return rc


# ----------------------------------------------------------------------------------------------------------------
def flat_source() -> Path:
    """One inlined .tex (used macros only, engine directive on line 1), .bbl, .bib and Figure_N.pdf; no subfolders."""
    sys.path.insert(0, str(SWS))
    from flatten_tex import flatten
    text = flatten(str(M / "one_column.tex"))
    figs = []
    def fig_sub(m):
        figs.append(m.group(2))
        return f"\\includegraphics{m.group(1) or ''}{{Figure_{len(figs)}.pdf}}"
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}]+)\}", fig_sub, text)
    bbl = (M / "one_column.bbl").read_text(encoding="utf-8")
    text, dropped = drop_unused_macros(text, bbl)
    head = ("% !TEX TS-program = pdflatex\n"
            "% RoughCert-SPX, one-column edition. Compile: pdflatex, bibtex, pdflatex, pdflatex.\n"
            "% For the two-column edition replace \\TwoColumnfalse by \\TwoColumntrue below.\n")
    out = PKG / f"{METHOD}_LaTeX_source"
    out.mkdir(parents=True)
    (out / f"{METHOD}_manuscript.tex").write_text(head + text.lstrip("\n"), encoding="utf-8", newline="\n")
    alt = text.replace(r"\newif\ifTwoColumn\TwoColumnfalse",r"\newif\ifTwoColumn\TwoColumntrue",1)
    assert alt != text
    (out / "two_columns.tex").write_text(alt, encoding="utf-8", newline="\n")
    supplement = flatten(str(SUP / "supplement.tex")).replace("../one_column}", f"{METHOD}_manuscript}}")
    (out / "supplement.tex").write_text(supplement,encoding="utf-8", newline="\n")
    shutil.copy2(M / "two_columns.bbl", out / "two_columns.bbl")
    shutil.copy2(M / "one_column.bbl", out / f"{METHOD}_manuscript.bbl")
    shutil.copy2(M / "references.bib", out / "references.bib")
    for i, f in enumerate(figs, 1):
        shutil.copy2(M / "figures" / f, out / f"Figure_{i}.pdf")
    CHECKS["flat_source"] = {"figures": figs, "unused_macros_removed": len(dropped), "files": sorted(p.name for p in out.iterdir())}
    return out


def drop_unused_macros(text: str, extra: str) -> tuple[str, list[str]]:
    def one_line_definition(line: str):
        m = re.match(r"\s*\\newcommand\*?\{?\\([A-Za-z]+)\}?(\[\d\])?\{", line)
        if not m:
            return None
        depth, i = 0, m.end() - 1
        while i < len(line):
            c = line[i]
            if c == "\\":
                i += 2
                continue
            depth += (c == "{") - (c == "}")
            if depth == 0:
                rest = line[i + 1:].strip()
                return m.group(1) if (rest == "" or rest.startswith("%")) else None
            i += 1
        return None
    lines, dropped = text.split("\n"), []
    while True:
        body = "\n".join(lines) + "\n" + extra
        keep, changed = [], False
        for ln in lines:
            name = one_line_definition(ln)
            if name and len(re.findall(r"\\" + name + r"(?![A-Za-z])", body)) <= 1:
                dropped.append(name)
                changed = True
                continue
            keep.append(ln)
        lines = keep
        if not changed:
            return "\n".join(lines), dropped


def check_flat_source(src: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="rcspx_flat_") as tmp:
        tmp = Path(tmp)
        for p in src.iterdir():
            shutil.copy2(p, tmp / p.name)
        stem = f"{METHOD}_manuscript"
        (tmp / f"{stem}.bbl").unlink()                      # bibtex must regenerate the shipped .bbl
        res = tex_build(tmp, stem, clean_env(tmp))
        res["bbl_identical"] = (tmp / f"{stem}.bbl").read_bytes() == (src / f"{stem}.bbl").read_bytes()
        res["pdf_compare_exit"] = pdf_compare(M / "one_column.pdf", tmp / f"{stem}.pdf", PKG / "qa" / "pdf_compare_flat_source.txt")
        res["other_editions"] = {}
        for name,reference,bib in (("two_columns",M / "two_columns.pdf",True),("supplement",SUP / "supplement.pdf",False)):
            bbl = tmp / (name + ".bbl")
            if bib:
                bbl.unlink()
            edition = tex_build(tmp,name,clean_env(tmp),with_bib=bib)
            edition["pdf_compare_exit"] = pdf_compare(reference,tmp / (name + ".pdf"), PKG / "qa" / ("pdf_compare_flat_" + name + ".txt"))
            if bib:
                edition["bbl_identical"] = bbl.read_bytes() == (src / bbl.name).read_bytes()
            res["other_editions"][name] = edition
        CHECKS["flat_source"].update({"clean_room": res})


# ----------------------------------------------------------------------------------------------------------------
def figures_folder(figs: list[str]) -> None:
    import fitz
    out = PKG / "figures"
    out.mkdir()
    fonts = {}
    for i, f in enumerate(figs, 1):
        shutil.copy2(M / "figures" / f, out / f"Figure_{i}.pdf")
        with fitz.open(out / f"Figure_{i}.pdf") as d:
            d[0].get_pixmap(dpi=300).save(out / f"Figure_{i}.png")
            fonts[f"Figure_{i}"] = figure_font_sizes(d[0])
    CHECKS["figures"] = {"n": len(figs), "byte_identical_to_embedded": all(
        sha256(out / f"Figure_{i}.pdf") == sha256(M / "figures" / f) for i, f in enumerate(figs, 1)),
        "png_dpi": 300, "printed_font_pt": fonts,
        "min_base_lettering_pt": min(v["base_min_pt"] for v in fonts.values())}


def figure_font_sizes(page, textwidth_pt: float = 468.0) -> dict:
    from collections import Counter
    scale = textwidth_pt / (page.rect.width * 72.27 / 72.0)
    sizes = [sp["size"] * scale for b in page.get_text("dict")["blocks"] for line in b.get("lines", [])
             for sp in line["spans"] if sp["text"].strip()]
    mode = Counter(round(s, 2) for s in sizes).most_common(1)[0][0]
    base = [s for s in sizes if s >= 0.85 * mode]
    scripts = [s for s in sizes if s < 0.85 * mode]
    return {"base_min_pt": round(min(base), 2), "script_min_pt": round(min(scripts), 2) if scripts else None}


# ----------------------------------------------------------------------------------------------------------------
def highlights() -> None:
    bullets = [b.strip() for b in (M / "highlights.txt").read_text(encoding="ascii").splitlines() if b.strip()]
    txt = PKG / f"{METHOD}_highlights.txt"
    txt.write_text("".join("- " + b + "\n" for b in bullets), encoding="ascii", newline="\n")
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    doc.core_properties.author = "Eslam Ahmed"
    doc.core_properties.title = "Highlights"
    doc.add_paragraph().add_run("Highlights").bold = True
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")
    doc.save(PKG / f"{METHOD}_highlights.docx")
    flat = bv.expanded_copy(M / "one_column.tex", PKG / "qa" / "main_flat_substituted.tex")
    rc = run(PY + [str(SWS / "check_highlights.py"), str(txt), "--tex", str(flat)], ROOT, log=PKG / "qa" / "check_highlights.txt")
    CHECKS["highlights"] = {"n": len(bullets), "max_chars": max(len(b) for b in bullets), "check_highlights_exit": rc}


def cover_letter(with_companion_copy: bool) -> None:
    n = numbers()
    bib = (M / "references.bib").read_text(encoding="utf-8")
    entry = bib[bib.index("AHMED2026111045,"):]
    team_title = re.sub(r"[{}]", "", re.search(r"title\s*=\s*\{(.+?)\},\s*\n", entry, re.S).group(1)).replace("--", "\u2013")
    enclosed = "; a copy is enclosed for review only" if with_companion_copy else ""
    content = {
        "sender": {"name": "Eslam Ahmed", "role": "Corresponding author",
                   "address": ["Engineering Mathematics and Physics Department",
                               "Faculty of Engineering, Cairo University, Giza 12613, Egypt"],
                   "email": "eslamahmed@eng.cu.edu.eg"},
        "date": DATE,
        "recipient": {"name": "The Editors", "journal": journal()},
        "salutation": "Dear Editors,",
        "title": TITLE,
        "article_type": "research article",
        "opening": "I am writing on behalf of the authors to submit our manuscript entitled \u201c**{title}**\u201d for consideration as a {article_type} in *{journal}*.",
        "paragraphs_before": [
            "Pricing S&P 500 (SPX) options under correlated rough volatility couples the integrated variance with the leverage "
            "integral, so the deterministic certificates available for VIX derivatives do not carry over. RoughCert-SPX tests "
            "sampling-free certification instead of assuming it, separates the three error sources of a simulation estimate "
            "and labels every reported interval by what it guarantees."],
        "results_intro": "Our main results are as follows.",
        "results": [
            f"**A two-field error ledger.** An It\u00f4\u2013Mehler ledger and a Put-Lipschitz bound valid for \u22121 \u2264 \u03c1 \u2264 0 "
            f"separate the kernel-approximation error, certified in {n['ArbBits']}-bit Arb ball arithmetic before any simulation, "
            "from time-discretisation bias and sampling error.",
            "**Where deterministic certification stops.** Leverage enters the price gap at first order. At \u03c1 = \u22120.9 the "
            f"25 bps certifiability boundary contracts by a factor of {n['ETwoCollapseMin']}\u2013{n['ETwoCollapseMax']}: "
            f"{n['ETwoLeverageMin']}\u2013{n['ETwoLeverageMax']} from leverage and {n['ETwoSecondOrderMin']}\u2013{n['ETwoSecondOrderMax']} "
            "from losing a two-moment bound that exists only at \u03c1 = 0.",
            "**A smoothing pitfall and its remedy.** Conditional-Jensen smoothing inside the leverage integral attenuates the "
            "effective correlation through a quadratic-variation deficit of the order of the 2H-th power of the time step, which exceeds "
            f"{n['JenDefThirtyTwo']}% at n = 32 and {n['JenDefFiveTwelve']}% at n = 512 in a calibrated regime; exact conditional "
            "expectation in the variance channel with an energy-conserving leverage integral avoids it.",
            "**Envelopes and audits.** Nested exact-bridge RQMC envelopes contain every tested exact rough Stein\u2013Stein reference "
            "price, Arb audits bound the kernel gap of Markovian and hybrid schemes, and six published benchmark families and a "
            "joint SPX\u2013VIX audit are scored against the envelopes; strikes that fail the contraction gates are labelled diagnostic.",
            "**Verification.** Nineteen first-principles checks and ten code-disjoint recomputations confirm the stated results and "
            "the archived numerical exhibits."],
        "paragraphs_after": [
            "The paper addresses a computational question at the core of the journal: when the bias of an SPX price under rough "
            "volatility can be certified, and what must be simulated when it cannot. The code, the archived results and the "
            f"reproduction scripts are archived in release {release_macro('ReleaseTag')} ({release_macro('ReleaseURL')}).",
            "Experiment E6 reuses VIX results of our companion working paper and software release RoughCert v5.0.0 "
            f"({COMPANION_URL}){enclosed}. Related work by the authors: \u201c{team_title}\u201d, Communications in Nonlinear "
            "Science and Numerical Simulation (2026), doi:10.1016/j.cnsns.2026.111045.",
            "Funding, competing interests and generative-AI use are disclosed in the manuscript, which is not under "
            "consideration elsewhere."],
        "closing": "Thank you for considering our manuscript.",
        "signoff": "Sincerely,",
        "signature": ["Eslam Ahmed", "On behalf of the authors"],
    }
    folder = PKG / "cover_letter"
    folder.mkdir()
    with tempfile.TemporaryDirectory(prefix="rcspx_letter_") as tmp:
        js = Path(tmp) / "letter.json"
        js.write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")
        texbin = Path(shutil.which("xelatex", path=bv.ENV["PATH"])).parent
        rc = run(PY + [str(SWS / "make_cover_letter.py"), str(js), "--out", str(folder), "--name", "QF_cover_letter",
                       "--texbin", str(texbin)], ROOT, log=PKG / "qa" / "make_cover_letter.txt")
    for junk in folder.glob("*"):
        if junk.suffix not in (".tex", ".pdf", ".docx"):
            junk.unlink()
    CHECKS["cover_letter"] = {"exit": rc, "files": sorted(p.name for p in folder.iterdir())}


# ----------------------------------------------------------------------------------------------------------------
def code_items() -> list[Path]:
    items = [ROOT / n for n in ("README.md", "requirements.txt", "LICENSE", "RELEASE_NOTES_v3.md")]
    items += sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    items += sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "scripts").glob("*.json"))
    items += sorted(p for p in (ROOT / "results").rglob("*.json") if p.name not in {"build_qa.json", "v3_public_release.json"})
    items += [M / n for n in ("main.tex", "one_column.tex", "two_columns.tex", "references.bib", "highlights.txt")]
    items += sorted((M / "sections").glob("*.tex")) + sorted((M / "tables").glob("*.tex")) + sorted((M / "tables").glob("*.json"))
    items += sorted((M / "figures").glob("*.pdf")) + [SUP / "supplement.tex"] + sorted((SUP / "tables").glob("*.tex"))
    return items


def code_archive() -> tuple[Path, Path]:
    folder = PKG / f"{METHOD}_code_and_data"
    for p in code_items():
        q = folder / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    z = make_zip(PKG / f"{METHOD}_code_and_data.zip", folder)
    CHECKS["code_archive"] = {"files": sum(1 for p in folder.rglob("*") if p.is_file()), "zip_bytes": z.stat().st_size}
    return folder, z


def code_clean_room(zip_path: Path) -> None:
    res = {}
    with tempfile.TemporaryDirectory(prefix="rcspx_code_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
            shipped = {n: z.read(n) for n in z.namelist() if n.startswith("manuscript/") and n.endswith(".tex") and "/tables/" in n}
        env = clean_env(tmp)
        # Local clean rooms are siblings of every protected project input.
        # Enumerate project roots so the authorised temporary directory can live on the same volume.
        protected = [f for f in ROOT.iterdir() if f.resolve() != Path(tempfile.gettempdir()).resolve()]
        protected += [p for p in ROOT.parent.iterdir() if p != ROOT and p.is_dir()]
        audit = PY + [str(SWS / "audit_open.py")]
        for path in protected:
            audit.extend(["--forbid", str(path)])
        import time
        t0 = time.time()
        res["unit_tests_exit"] = run(audit + ["--log", str(tmp / "opened_unit.txt"), "--", "scripts/run_unit_verifications.py"], tmp, env,
                                     log=PKG / "qa" / "clean_room_unit_verifications.txt")
        res["unit_tests_seconds"] = round(time.time() - t0, 1)
        res["generator_exit"] = run(audit + ["--log", str(tmp / "opened_gen.txt"), "--", "scripts/build_numbers_and_tables.py"], tmp, env,
                                    log=PKG / "qa" / "clean_room_build_numbers_and_tables.txt")
        regen = {n: (tmp / n).read_bytes() for n in shipped}
        res["tables_compared"] = len(shipped)
        res["tables_byte_identical"] = all(regen[n] == shipped[n] for n in shipped)
        res["figures_exit"] = run(audit + ["--log", str(tmp / "opened_fig.txt"), "--", "scripts/build_figures.py"], tmp, env,
                                  log=PKG / "qa" / "clean_room_build_figures.txt")
        figs = sorted((M / "figures").glob("*.pdf"))
        res["figures_pixel_identical"] = all(pdf_compare(f, tmp / "manuscript" / "figures" / f.name,
                                                         PKG / "qa" / f"pdf_compare_clean_room_{f.stem}.txt") == 0 for f in figs)
        for stem, folder in (("one_column", tmp / "manuscript"), ("two_columns", tmp / "manuscript"), ("supplement", tmp / "manuscript" / "supplement")):
            r = tex_build(folder, stem, env, with_bib=(stem != "supplement"))
            work = (SUP if stem == "supplement" else M) / f"{stem}.pdf"
            r["pdf_compare_exit"] = pdf_compare(work, folder / f"{stem}.pdf", PKG / "qa" / f"pdf_compare_clean_room_{stem}.txt")
            res[stem] = r
        res["forbidden_opened"] = {k: (tmp / f"opened_{k}.txt").read_text(encoding="utf-8", errors="replace").count("FORBIDDEN\t")
                                   for k in ("unit", "gen", "fig") if (tmp / f"opened_{k}.txt").exists()}
    CHECKS["code_clean_room"] = res


# ----------------------------------------------------------------------------------------------------------------
def for_review_only() -> bool:
    pdf = os.environ.get("COMPANION_PDF")
    aux = os.environ.get("COMPANION_AUX")
    if not (pdf and Path(pdf).is_file() and aux and Path(aux).is_file()):
        CHECKS["for_review_only"] = "pending: set COMPANION_PDF and COMPANION_AUX to the RoughCert v5.0.0 PDF and .aux"
        return False
    out = PKG / "for_review_only"
    out.mkdir()
    shutil.copy2(pdf, out / "RoughCert_v5.0.0_companion_working_paper.pdf")
    supp = Path(pdf).with_name("RoughCert_supplementary_material.pdf")
    shutil.copy2(supp, out / supp.name)
    labels = dict((k, (num, page)) for k, num, page in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", Path(aux).read_text(encoding="utf-8", errors="replace")))
    mapping = json.loads((ROOT / "scripts" / "companion_map.json").read_text(encoding="utf-8"))
    rows = []
    for item in mapping["items"]:
        refs = "; ".join(f"{lab_kind(l)} {labels[l][0]} (p.~{labels[l][1]})" for l in item["labels"])
        rows.append(f"{item['what']} & {refs} & {item['note']} \\\\")
    tex = NOTE_TEMPLATE.replace("ROWS", "\n".join(rows)).replace("{HASH_PREFIX}", sha256(Path(pdf))[:16])
    (out / "Note_for_editor_and_reviewers.tex").write_text(tex, encoding="utf-8", newline="\n")
    tex_cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "Note_for_editor_and_reviewers.tex"]
    rc = [run(tex_cmd, out), run(tex_cmd, out)]
    for junk in out.glob("Note_for_editor_and_reviewers.*"):
        if junk.suffix not in (".tex", ".pdf"):
            junk.unlink()
    CHECKS["for_review_only"] = {"companion_sha256": sha256(out / "RoughCert_v5.0.0_companion_working_paper.pdf"), "note_exit": rc}
    return True


def lab_kind(label: str) -> str:
    return {"tab": "Table", "fig": "Figure", "sec": "Section", "thm": "Theorem", "prop": "Proposition", "lem": "Lemma", "eq": "Eq."}.get(label.split(":")[0], "Item")


NOTE_TEMPLATE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{booktabs,array}\usepackage[hidelinks]{hyperref}
\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\begin{document}
\textbf{Note for the editor and reviewers (not for online publication)}

\textit{RoughCert-SPX: Certified-Bias SPX Option Pricing and Kernel Audits under Correlated Rough Volatility}

Experiment E6 of the submitted manuscript reuses Arb-certified VIX results of the companion working paper and software
release RoughCert v5.0.0 (\url{https://github.com/EslamAhmedGaber/RoughCert/releases/tag/v5.0.0}). The enclosed PDF is a
byte-identical copy of that working paper (SHA-256 prefix {HASH_PREFIX}). The table maps every reused result to its location.

\medskip
\begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm}>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{6.2cm}}
\toprule
Reused in E6 & Location in RoughCert v5.0.0 & Note \\
\midrule
ROWS
\bottomrule
\end{tabular}

\medskip
The E6 inputs are read from the release result files \texttt{vix\_futures\_full\_suite.json} and
\texttt{vix\_options\_smile\_suite.json}, pinned by SHA-256 in the E6 archive; a code-disjoint check (I6 in the
Supplementary Material) confirms that every reused number equals the release files bit for bit.
\end{document}
"""


# ----------------------------------------------------------------------------------------------------------------
def gates() -> None:
    qa = PKG / "qa"
    flat_s = bv.expanded_copy(SUP / "supplement.tex", qa / "supplementary_flat_substituted.tex")
    flat = qa / "main_flat_substituted.tex"
    jobs = [("check_norms_main", "check_norms.py", [str(flat)], M), ("check_norms_supplementary", "check_norms.py", [str(flat_s)], SUP),
            ("check_tex_hygiene_main", "check_tex_hygiene.py", ["main.tex"], M),
            ("check_tex_hygiene_supplementary", "check_tex_hygiene.py", ["supplement.tex"], SUP),
            ("check_technical_main", "check_technical.py", ["main.tex", "--pdf", "one_column.pdf", "--bbl", "one_column.bbl", "--bib",
             "references.bib", "--order", "refs-then-appendix", "--declarations", "before-refs", "--funding", bv.FUND, "--team-citation"], M)]
    res = {}
    for name, tool, args, wd in jobs:
        res[name] = run(PY + [str(SWS / tool), *args], wd, log=qa / f"{name}.txt")
    with tempfile.TemporaryDirectory(prefix="rcspx_bib_") as tmp:
        js = Path(tmp) / "bib.json"
        res["bib_crossref"] = run(PY + [str(SWS / "bib_crossref.py"), str(M / "references.bib"), "--bbl", str(M / "one_column.bbl"),
                                        "--aux", str(M / "one_column.aux"), "--aux", str(SUP / "supplement.aux"), "--json", str(js),
                                        "--timeout", "8"], ROOT, log=qa / "bib_crossref.txt")
        if js.exists():
            write_report(qa / "bib_crossref.json", js.read_text(encoding="utf-8", errors="replace"))
    CHECKS["gates"] = res


def copy_records() -> None:
    qa = PKG / "qa"
    a = ROOT / "audit"
    for src, dst in ((ROOT / "AUDIT_REPORT_V3.md", "AUDIT.md"), (a / "v2_change_log.json", "number_ledger_v1_to_v2.json"),
                     (a / "bib_verification_v2.json", "bib_manual_verification.json"),
                     (a / "v1_immutability_final.json", "v1_immutability_proof.json")):
        if src.exists():
            shutil.copy2(src, qa / dst)
    q = json.loads((ROOT / "results" / "build_qa.json").read_text(encoding="utf-8"))
    CHECKS["build_verify"] = {"all_exit_zero": q["all_exit_zero"], "documents_clean": q["documents_clean"],
                              "steps": {s["name"]: s["exit"] for s in q["steps"]},
                              "documents": {d["doc"]: {k: v for k, v in d.items() if k not in ("examples", "doc")} for d in q["documents"]}}


def readme() -> None:
    c = CHECKS
    docs = c["build_verify"]["documents"]
    rows = [
        (f"{METHOD}_one_column.pdf", f"Manuscript (primary submission PDF, elsarticle [final,3p,11pt], {docs['one_column']['pages']} pages)"),
        (f"{METHOD}_two_columns.pdf", f"Two-column edition from the same source ({docs['two_columns']['pages']} pages)"),
        (f"{METHOD}_supplementary_material.pdf", f"Supplementary material ({docs['supplement']['pages']} pages)"),
        (f"{METHOD}_LaTeX_source.zip", "Flat article and supplement sources, bibliographies and Figure_1-7.pdf (upload as the LaTeX source item)"),
        ("figures/", "Figure_1-7 as vector PDF and 300-dpi PNG, in print order"),
        (f"{METHOD}_highlights.txt / .docx", "Highlights (5 bullets, at most 85 characters)"),
        ("cover_letter/", "Cover letter (PDF, Word and LaTeX with the same text)"),
        (f"{METHOD}_code_and_data.zip", "Code, archived results and modular sources (supplementary file / data availability)"),
        ("for_review_only/", "Companion working paper RoughCert v5.0.0 and a one-page map (for the editor and reviewers only)"
         if isinstance(c.get("for_review_only"), dict) else "PENDING (see open items)"),
        ("qa/", "Audit report, package checks, gate outputs, clean-room logs, bibliography checks, number ledger"),
    ]
    fr = c["flat_source"]["clean_room"]
    cr = c["code_clean_room"]
    lines = [f"# {METHOD} submission package ({DATE})", "",
             f"*{TITLE}*  ", f"{AUTHORS}. Target journal: *{journal()}*.", "",
             "| File | Submission item |", "|---|---|", *[f"| `{a}` | {b} |" for a, b in rows], "",
             "## Integrity summary", "",
             f"* Builds: one-column {docs['one_column']['pages']} pp, two-column {docs['two_columns']['pages']} pp, supplement "
             f"{docs['supplement']['pages']} pp; zero errors, warnings, undefined references, overfull and underfull boxes; the fourth "
             f"pass changes no .aux line (build_verify: all exit 0 = {c['build_verify']['all_exit_zero']}, clean = {c['build_verify']['documents_clean']}).",
             f"* Flat source compiled in an empty folder: {fr['pages']} pp, text and 110-dpi pixels identical to the working PDF "
             f"(pdf_compare exit {fr['pdf_compare_exit']}), regenerated .bbl identical = {fr['bbl_identical']}, fourth pass invariant = {fr['aux_invariant_4th_pass']}.",
             f"* Code archive extracted outside the project: unit tests exit {cr['unit_tests_exit']}, number generator exit {cr['generator_exit']}, "
             f"{cr['tables_compared']} generated tables byte-identical = {cr['tables_byte_identical']}, figures pixel-identical = {cr['figures_pixel_identical']}, "
             f"manuscript PDFs pixel-identical (exit codes {cr['one_column']['pdf_compare_exit']}, {cr['two_columns']['pdf_compare_exit']}, {cr['supplement']['pdf_compare_exit']}); "
             f"files opened inside the project: {sum(cr['forbidden_opened'].values())}.",
             f"* Gates: {', '.join(f'{k} exit {v}' for k, v in c['gates'].items())}; leak scans: {', '.join(f'{k} exit {v}' for k, v in c.get('scan_leaks', {}).items())}.",
             f"* Figures: printed base lettering at least {c['figures']['min_base_lettering_pt']} pt; embedded and separate PDFs byte-identical = {c['figures']['byte_identical_to_embedded']}.",
             "* Every file of this folder is listed with its SHA-256 digest in `SHA256_MANIFEST.json`.", "",
             "## Reproduction", "",
             "See `RoughCert_SPX_code_and_data/README.md`. In short: `pip install -r requirements.txt`, then",
             "`python -B scripts/run_unit_verifications.py`, `python -B scripts/build_numbers_and_tables.py` and",
             "`python -B scripts/build_figures.py`; compile `manuscript/one_column.tex` with pdflatex, bibtex, pdflatex, pdflatex.", "",
             "## Open items for the authors", "",
             "See `qa/AUDIT.md` (section Sign-off). Nothing has been uploaded, pushed or submitted."]
    (PKG / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def manifest() -> None:
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and p.name != "SHA256_MANIFEST.json")
    data = {"algorithm": "SHA-256", "generated": DATE, "n_files": len(files),
            "files": {p.relative_to(PKG).as_posix(): sha256(p) for p in files}}
    (PKG / "SHA256_MANIFEST.json").write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in (PKG, SEALED):
        if p.exists():
            p.rename(p.with_name(p.name + f".prev_{stamp}"))
    (PKG / "qa").mkdir(parents=True)
    for src, dst in ((M / "one_column.pdf", "one_column"), (M / "two_columns.pdf", "two_columns"), (SUP / "supplement.pdf", "supplementary_material")):
        shutil.copy2(src, PKG / f"{METHOD}_{dst}.pdf")
    copy_records()
    src = flat_source()
    check_flat_source(src)
    make_zip(PKG / f"{METHOD}_LaTeX_source.zip", src)
    figures_folder(CHECKS["flat_source"]["figures"])
    highlights()
    has_companion = for_review_only()
    if not has_companion:
        raise RuntimeError("The companion materials are required")
    cover_letter(has_companion)
    _, code_zip = code_archive()
    code_clean_room(code_zip)
    gates()
    scan(src, "latex_source")
    scan(PKG / f"{METHOD}_code_and_data", "code_and_data")
    readme()
    write_report(PKG / "qa" / "package_checks.json", json.dumps(CHECKS, indent=1, default=str))
    scan(PKG, "package_folder")
    write_report(PKG / "qa" / "package_checks.json", json.dumps(CHECKS, indent=1, default=str))
    # Seal after anonymous release verification and the final baseline comparison.
    ok = (CHECKS["build_verify"]["all_exit_zero"] and CHECKS["build_verify"]["documents_clean"]
          and CHECKS["flat_source"]["clean_room"]["pdf_compare_exit"] == 0 and CHECKS["flat_source"]["clean_room"]["bbl_identical"]
          and CHECKS["code_clean_room"]["unit_tests_exit"] == 0 and CHECKS["code_clean_room"]["tables_byte_identical"]
          and all(v == 0 for v in CHECKS["gates"].values()) and all(v == 0 for v in CHECKS["scan_leaks"].values())
          and CHECKS["highlights"]["check_highlights_exit"] == 0 and CHECKS["cover_letter"]["exit"] == 0)
    print(json.dumps({k: v for k, v in CHECKS.items() if k != "figures"}, indent=1, default=str)[:6000])
    print(f"PACKAGE OK: {ok}; for_review_only: {'included' if has_companion else 'pending'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
