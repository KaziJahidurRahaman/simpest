#!/usr/bin/env python
"""Matched dual run: C# FraNchEstYN vs Python simpest on identical inputs.

Purpose
-------
Run the reference C# FraNchEstYN and the Python ``simpest`` franchestyn step on
*the same* weather, sowing, external crop-model, reference, and parameter files,
then diff the daily outputs column-by-column. This converts the structural
replication review into a measured numeric tolerance.

Design
------
* Both engines read identical input files (assembled under ``inputs/``). The
  parameter file is the *same* ``franchestynParameters.csv`` for both, so
  parameter values are guaranteed equal (Python's legacy CSV reader uses the
  same ``Name_Class`` last-row-wins keying as C#).
* The Python side is driven through ``FranchestynRunner`` directly (not the
  SIMPLACE/JVM pipeline), with the C#-parity flags chosen in the replication
  fixes: ``use_gdd=False`` (calendar cycle %) and
  ``use_prev_day_alignment=True`` (C# objective day alignment). Disease is
  enabled; fungicide is off (the Indiana inputs schedule no treatments).
* The run is a *validation* run (no calibration) so the comparison isolates
  model logic, not the stochastic optimizer.

Caveats
-------
* The C# binary under ``bin/Debug/net8.0`` is a prebuilt artifact and may be an
  older build than the head C# source. The .NET SDK is not on PATH here, so it
  cannot be rebuilt from source automatically.
* Columns ``DaysAfterSowing`` and ``GrowingDegreeDays`` are expected to differ
  in the external-crop branch (finding F7: Python populates them, C# leaves 0).

Usage
-----
    python tools/dual_run/run_dual.py            # assemble + run both + diff
    python tools/dual_run/run_dual.py --python-only
    python tools/dual_run/run_dual.py --no-assemble
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths and run settings
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EX = REPO / "docs" / "examples" / "data"
CS_DIR = REPO / "franchestyn_ref" / "src_csharp" / "FraNchEstYN" / "FraNchEstYN" / "bin" / "Debug" / "net8.0"
CS_EXE = CS_DIR / "FraNchEstYN.exe"

INPUTS = HERE / "inputs"
OUT = HERE / "out"
CONFIG_PATH = HERE / "config_indiana.json"

SITE = "indiana"
VARIETY = "Generic"
DISEASE = "thisDisease"
START_YEAR = 1973
END_YEAR = 1990

# Source files from the Python example data, mapped to the shared input layout
# that both readers expect.
SRC = EX / "simpest_outputs" / "SimulationExperimentTemplate"
FILE_MAP = {
    SRC / "weather_franchestyn.csv": INPUTS / "weather" / "daily" / f"{SITE}.csv",
    SRC / "cropModel_data.csv": INPUTS / "cropModel" / "cropModelData.csv",
    SRC / "management_franchestyn.csv": INPUTS / "management" / "sowing.csv",
    EX / "reference_fra" / "FRA_REFERENCE_Indiana.csv": INPUTS / "reference" / "referenceData.csv",
    CS_DIR / "files" / "parameters" / "franchestynParameters.csv": INPUTS / "parameters" / "franchestynParameters.csv",
}


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def assemble_inputs() -> None:
    """Copy the matched inputs into the shared layout consumed by both engines."""
    for src, dst in FILE_MAP.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source input: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  copied {src.name} -> {dst.relative_to(HERE)}")


def write_cs_config() -> None:
    """Write the C# config JSON pointing at the shared inputs (validation mode)."""
    config = {
        "settings": {
            "startYear": START_YEAR,
            "endYear": END_YEAR,
            "sites": [SITE],
            "isCalibration": "false",
            "calibrationVariable": "all",
            "varieties": [VARIETY],
            "cropModel": "yes",
            "simplexes": 1,
            "disease": DISEASE,
            "iterations": 1,
            "weatherTimeStep": "daily",
        },
        "paths": {
            "weatherDir": str(INPUTS / "weather"),
            "referenceFilePaths": str(INPUTS / "reference"),
            "paramFile": str(INPUTS / "parameters" / "franchestynParameters.csv"),
            "sowingFile": str(INPUTS / "management" / "sowing.csv"),
            "cropModelFile": str(INPUTS / "cropModel"),
        },
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  wrote {CONFIG_PATH.relative_to(HERE)}")


def run_csharp() -> Path | None:
    """Run the prebuilt C# exe; return the path to its daily output CSV."""
    if not CS_EXE.exists():
        print(f"  [skip] C# exe not found at {CS_EXE}")
        return None
    cs_out = CS_DIR / "outputs" / f"{SITE}_{VARIETY}.csv"
    if cs_out.exists():
        cs_out.unlink()
    print(f"  running: {CS_EXE.name} {CONFIG_PATH.name}")
    proc = subprocess.run(
        [str(CS_EXE), str(CONFIG_PATH)],
        cwd=str(CS_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    for line in tail:
        print(f"    C#> {line}")
    if proc.returncode != 0:
        print(f"  [warn] C# exit code {proc.returncode}; stderr:\n{proc.stderr[:1000]}")
    if not cs_out.exists():
        print("  [warn] C# produced no output file")
        return None
    dst = OUT / "csharp_daily.csv"
    shutil.copyfile(cs_out, dst)
    print(f"  C# output -> {dst.relative_to(HERE)} ({sum(1 for _ in dst.open())-1} rows)")
    return dst


def run_python() -> Path:
    """Run the Python franchestyn step on the matched inputs; write daily CSV."""
    sys.path.insert(0, str(REPO))
    from simpest.models.fr_runner import FranchestynRunner
    from simpest.models.franchestyn import _outputs_to_records

    runner = FranchestynRunner(
        weather_dir=str(INPUTS / "weather" / "daily" / f"{SITE}.csv"),
        param_file=str(INPUTS / "parameters" / "franchestynParameters.csv"),
        sowing_file=str(INPUTS / "management" / "sowing.csv"),
        ref_dir=str(INPUTS / "reference"),
        crop_model_dir=str(INPUTS / "cropModel"),
        site=SITE,
        variety=VARIETY,
        disease=DISEASE,
        start_year=START_YEAR,
        end_year=END_YEAR,
        weather_time_step="daily",
        calibration_variable="all",
        is_calibration=False,
        crop_type=None,            # -> legacy CSV parameter path (parity with C#)
        disease_type=DISEASE,      # truthy -> enable disease model
        fungicide_type=None,       # no fungicide treatments in the Indiana inputs
        use_gdd=False,             # F2: calendar cycle % (C# parity)
        use_prev_day_alignment=True,  # F3: C# objective day alignment
    )
    date_outputs = runner.run()
    df = pd.DataFrame(_outputs_to_records(date_outputs))
    dst = OUT / "python_daily.csv"
    df.to_csv(dst, index=False)
    print(f"  Python output -> {dst.relative_to(HERE)} ({len(df)} rows)")
    return dst


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
# (label, [C# column aliases], python column)
COMPARE = [
    ("CyclePct",       ["cyclePerc", "CyclePercentage"],        "CycleCompletionPercentage"),
    ("LightIntAtt",    ["lightInt", "LightInterception"],       "LightInterception"),
    ("LightIntHealthy",["lightIntHealthy", "LightIntHealthy"],  "LightIntHealthy"),
    ("AGBatt",         ["agb", "AGBattainable"],                "AGBattainable"),
    ("AGBact",         ["agbHealthy", "AGBactual"],             "AGBactual"),
    ("YieldAtt",       ["yield", "YieldAttainable"],            "YieldAttainable"),
    ("YieldAct",       ["yieldHealthy", "YieldActual"],         "YieldActual"),
    ("HTtimeR",        ["htTimeR", "HTtimeRinoculum"],          "HTtimeRinoculum"),
    ("HTtimeS",        ["htTimeS", "HTtimeSinoculum"],          "HTtimeSinoculum"),
    ("Susceptible",    ["susceptible", "Susceptible"],          "Susceptible"),
    ("Latent",         ["latent", "Latent"],                    "Latent"),
    ("Sporulating",    ["sporulating", "Sporulating"],          "Sporulating"),
    ("Dead",           ["dead", "Dead"],                        "Dead"),
    ("Affected",       ["affected", "Affected"],                "Affected"),
    ("DiseaseSev",     ["disSev", "DiseaseSeverity"],           "DiseaseSeverity"),
    ("Tmax",           ["Tmax"],                                "Tmax"),
    ("Tmin",           ["Tmin"],                                "Tmin"),
    ("Prec",           ["Prec"],                                "TotalPrec"),
    ("Rad",            ["Rad"],                                 "TotalRad"),
    # Expected to differ by design (finding F7):
    ("DaysAfterSowing*", ["das", "DaysAfterSowing"],            "DaysAfterSowing"),
    ("GDD*",           ["gdd", "GrowingDegreeDays"],            "GrowingDegreeDays"),
]


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _year_doy_key(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    if kind == "cs":
        ycol = _first_present(df, ["year", "Year"])
        dcol = _first_present(df, ["doy", "Doy"])
        df["_year"] = pd.to_numeric(df[ycol], errors="coerce")
        df["_doy"] = pd.to_numeric(df[dcol], errors="coerce")
    else:  # python: parse Date "dd/mm/YYYY"
        dt = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["_year"] = dt.dt.year
        df["_doy"] = dt.dt.dayofyear
    return df.dropna(subset=["_year", "_doy"])


def compare(cs_path: Path, py_path: Path) -> str:
    cs_raw = pd.read_csv(cs_path)
    cs_raw.columns = cs_raw.columns.str.strip()  # C# header has leading-space cols
    cs = _year_doy_key(cs_raw, "cs")
    py = _year_doy_key(pd.read_csv(py_path), "py")
    merged = pd.merge(cs, py, on=["_year", "_doy"], suffixes=("_cs", "_py"))

    lines = []
    lines.append("=" * 78)
    lines.append("MATCHED DUAL RUN — C# FraNchEstYN vs Python simpest (Indiana, validation)")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"years: {START_YEAR}-{END_YEAR}   site: {SITE}   variety: {VARIETY}")
    lines.append(f"C# rows: {len(cs)}   Python rows: {len(py)}   matched (year,doy): {len(merged)}")
    lines.append("flags: use_gdd=False, use_prev_day_alignment=True, disease=on, fungicide=off")
    lines.append("=" * 78)
    hdr = f"{'column':<18}{'n':>6}{'max|diff|':>14}{'mean|diff|':>14}{'corr':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for label, cs_aliases, py_col in COMPARE:
        cs_col = _first_present(cs, cs_aliases)
        if cs_col is None or py_col not in py.columns:
            lines.append(f"{label:<18}{'--':>6}   (missing column)")
            continue
        a = pd.to_numeric(merged[f"{cs_col}_cs"] if f"{cs_col}_cs" in merged else merged[cs_col], errors="coerce")
        b = pd.to_numeric(merged[f"{py_col}_py"] if f"{py_col}_py" in merged else merged[py_col], errors="coerce")
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        if len(a) == 0:
            lines.append(f"{label:<18}{0:>6}   (no overlapping values)")
            continue
        diff = (a - b).abs()
        corr = a.corr(b) if a.std() > 0 and b.std() > 0 else float("nan")
        lines.append(f"{label:<18}{len(a):>6}{diff.max():>14.6g}{diff.mean():>14.6g}{corr:>8.4f}")

    lines.append("-" * len(hdr))
    lines.append("* DaysAfterSowing / GDD differ by design in the external-crop branch (F7).")
    report = "\n".join(lines)
    (OUT / "diff_report.txt").write_text(report, encoding="utf-8")
    return report


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-assemble", action="store_true", help="skip copying inputs")
    ap.add_argument("--python-only", action="store_true", help="skip the C# run")
    args = ap.parse_args()

    print("[1/4] assemble inputs")
    if not args.no_assemble:
        assemble_inputs()
    print("[2/4] write C# config")
    write_cs_config()
    print("[3/4] run engines")
    cs_path = None if args.python_only else run_csharp()
    py_path = run_python()
    print("[4/4] compare")
    if cs_path is None:
        print("  C# output unavailable — wrote Python output only; skipping diff.")
        return
    report = compare(cs_path, py_path)
    print()
    print(report)


if __name__ == "__main__":
    main()
