"""Replace the machine-specific path in results/e6_joint_spx_vix.json (field paper1_origin) by a
path-free provenance string that pins the two RoughCert v5.0.0 input files by SHA-256.

Only that one JSON string is rewritten (exact text replacement); every number in the archive stays
byte-identical. Usage: python -B scripts/sanitize_e6_origin.py [--paper1-dir <RoughCert v5.0.0 results dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "results" / "e6_joint_spx_vix.json"
NAMES = ("vix_futures_full_suite.json", "vix_options_smile_suite.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper1-dir", type=Path, default=ROOT / "results" / "companion_v5")
    args = ap.parse_args()
    raw = TARGET.read_text(encoding="utf-8")
    data = json.loads(raw)
    old = data["paper1_origin"]
    if not old.startswith("live_read_only:"):
        print("paper1_origin already path-free:", old[:120])
        return
    digests = {n: hashlib.sha256((args.paper1_dir / n).read_bytes()).hexdigest() for n in NAMES}
    new = "read_only:RoughCert v5.0.0 results/" + "; ".join(f"{n} sha256={d}" for n, d in digests.items())
    enc_old, enc_new = json.dumps(old), json.dumps(new)
    if raw.count(enc_old) != 1:
        raise SystemExit("FATAL: paper1_origin string not found exactly once")
    out = raw.replace(enc_old, enc_new)
    check = json.loads(out)
    data["paper1_origin"] = new
    if check != data:
        raise SystemExit("FATAL: sanitized archive differs beyond paper1_origin")
    TARGET.write_text(out, encoding="utf-8")
    print("paper1_origin ->", new)


if __name__ == "__main__":
    main()
