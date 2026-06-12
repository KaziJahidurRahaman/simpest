#!/usr/bin/env python
"""Verify simpest's JSON parameter files match the canonical FraNchEstYN source.

The authoritative per-crop / per-disease / per-fungicide parameter database in
FraNchEstYN is ``franchestynParametersForPackage.csv`` (columns
``Model,Set,Parameter,...,min,max,value,calibration``) — the same file that
builds the R package datasets. simpest's JSON files
(``fr_{crop,disease,fungicide}_parameters.json``) should match it exactly.

This is a regression guard for F10: it compares every (model, set, parameter)
for ``min``, ``max``, ``value`` and the ``calibration`` flag, and reports any
drift. Exits non-zero if mismatches are found.

Note: the C# *runtime* file ``franchestynParameters.csv`` (concatenated, no Set
column, last-block-wins) is a flattened artifact that uses blanket bounds for
some parameters (e.g. ``AssimilateSappersDamage`` 0-300 across all diseases) and
is NOT the canonical per-disease source — do not compare against it.

Usage:
    python tools/check_parameters.py [path/to/franchestynParametersForPackage.csv]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "simpest" / "models"
DEFAULT_REF = (
    REPO / "franchestyn_ref" / "src_csharp" / "FraNchEstYN" / "FraNchEstYN"
    / "files" / "parameters" / "franchestynParametersForPackage.csv"
)
JSON_FILES = {
    "crop": MODELS / "fr_crop_parameters.json",
    "disease": MODELS / "fr_disease_parameters.json",
    "fungicide": MODELS / "fr_fungicide_parameters.json",
}
_TRUE = {"x", "true", "1", "yes"}


def _fnum(s):
    s = ("" if s is None else str(s)).strip()
    try:
        return float(s)
    except ValueError:
        return None


def _load_reference(path: Path) -> dict:
    ref: dict = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            m = r["Model"].strip().lower()
            s = r["Set"].strip()
            p = r["Parameter"].strip()
            ref.setdefault(m, {}).setdefault(s, {})[p] = {
                "min": _fnum(r["min"]),
                "max": _fnum(r["max"]),
                "value": _fnum(r["value"]),
                "cal": (r["calibration"] or "").strip().lower() in _TRUE,
            }
    return ref


def _load_json() -> dict:
    jsn: dict = {}
    for model, f in JSON_FILES.items():
        data = json.loads(f.read_text(encoding="utf-8"))
        for s, params in data.items():
            for p, spec in params.items():
                jsn.setdefault(model, {}).setdefault(s, {})[p] = {
                    "min": _fnum(spec.get("min")),
                    "max": _fnum(spec.get("max")),
                    "value": _fnum(spec.get("value")),
                    "cal": bool(spec.get("calibration", False)),
                }
    return jsn


def _close(a, b) -> bool:
    if a is None or b is None:
        return a == b
    return abs(a - b) < 1e-9


def main() -> int:
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF
    if not ref_path.exists():
        print(f"[skip] canonical reference not found: {ref_path}")
        print("       (clone GeoModelLab/FraNchEstYN to franchestyn_ref/ or pass the path)")
        return 0

    ref, jsn = _load_reference(ref_path), _load_json()
    mismatches, total = [], 0
    for m in ref:
        for s in ref[m]:
            jset = jsn.get(m, {}).get(s, {})
            for p, rv in ref[m][s].items():
                total += 1
                jv = jset.get(p)
                if jv is None:
                    mismatches.append(f"[missing in JSON] {m}/{s}/{p}")
                    continue
                diffs = [k for k in ("min", "max", "value") if not _close(rv[k], jv[k])]
                if rv["cal"] != jv["cal"]:
                    diffs.append("cal")
                if diffs:
                    det = ", ".join(f"{k}: ref={rv[k]} json={jv[k]}" for k in diffs)
                    mismatches.append(f"{m}/{s}/{p}: {det}")
            for p in jset:
                if p not in ref[m][s]:
                    mismatches.append(f"[extra in JSON] {m}/{s}/{p}")

    for line in mismatches:
        print(line)
    print(f"\n{total} parameters compared, {len(mismatches)} mismatch(es) "
          f"vs {ref_path.name}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
