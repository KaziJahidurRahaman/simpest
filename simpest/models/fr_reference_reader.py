"""Readers for sowing schedules, reference observations, and crop-model series.

This module loads the non-weather inputs that drive and constrain a run:

- the per-site sowing schedule and fungicide treatment dates,
- field reference observations used to score the model during calibration, and
- a daily crop-model series (light interception, biomass, yield, and optional
  thermal time) supplied by an external crop growth model.

Each reader populates the appropriate fields of a
:class:`~simpest.models.fr_data.SimulationUnit` or returns a
:class:`~simpest.models.fr_data.CropModelData` container.
"""

from __future__ import annotations

import csv
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .fr_data import CropModelData, ReferenceData, SimulationUnit, FungicideTreatmentSchedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize a column name: lowercase, strip spaces/underscores/quotes."""
    return "".join(
        ch for ch in (s or "").strip().strip('"') if ch not in (" ", "_")
    ).lower()


def _get_col(header_map: dict, *aliases: str) -> int:
    """Return column index for the first matching alias, or -1 if not found."""
    for a in aliases:
        key = _norm(a)
        if key in header_map:
            return header_map[key]
    return -1


def _pf(s: str) -> Optional[float]:
    """Parse a float; returns None on failure or 'NA'."""
    s = (s or "").strip().strip('"')
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_sowing(
    sowing_file: str | Path,
    site: str,
    variety: str,
    start_year: int,
    end_year: int,
    all_row_includes_end_year: bool = False,
) -> SimulationUnit:
    """Read sowing.csv and build the SimulationUnit for one site × variety.

    Expected CSV columns (case-insensitive):
        site, crop, variety, sowingDOY, year
        (+ optional treatment1, treatment2, ... columns for fungicide dates)

    Args:
        sowing_file: Path to the sowing CSV.
        site:        Site identifier to filter on (e.g., "indiana").
        variety:     Variety identifier to filter on (e.g., "Generic").
        start_year:  First simulation year (inclusive).
        end_year:    Last simulation year (inclusive).
        all_row_includes_end_year: How far an ``"All"`` sowing row is applied.
            When ``False`` (default) the ``"All"`` row is applied to the half-open
            range ``[start_year, end_year - 1]`` and therefore omits the final
            year; when ``True`` it is applied through ``end_year`` inclusive. This
            setting affects only ``"All"`` rows; explicit per-year rows are always
            honoured for every year, so it has no effect when sowing is specified
            per year (for example the CSV written by
            :func:`simpest.models.simplace.build_management`).

    Returns:
        Populated SimulationUnit (year_sowing_doy, fungicide_treatment_schedule).
    """
    sim = SimulationUnit()
    path = Path(sowing_file)
    # print(f'sowing_file: {sowing_file}')

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        raw_headers = next(reader, [])
        headers = [h.strip().strip('"').lower().replace(" ", "").replace("_", "") for h in raw_headers]
        header_map = {h: i for i, h in enumerate(headers)}

        site_idx = _get_col(header_map, "site")
        crop_idx = _get_col(header_map, "crop")
        variety_idx = _get_col(header_map, "variety", "cultivar", "cv")
        sowing_idx = _get_col(header_map, "sowingdoy", "sowingday", "doy")
        year_idx = _get_col(header_map, "year", "yr")

        # Collect treatment columns (any column starting with "treatment")
        fung_indices = [i for i, h in enumerate(headers) if h.startswith("treatment")]

        # Store "All" row separately; per-year rows in a dict
        all_row: Optional[Tuple[int, List[int]]] = None
        per_year: Dict[int, Tuple[int, List[int]]] = {}

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            if site_idx >= 0:
                row_site = row[site_idx].strip().strip('"').lower()
                if row_site != site.lower():
                    continue

            if variety_idx >= 0:
                row_var = row[variety_idx].strip().strip('"').lower()
                if row_var != variety.lower():
                    continue

            if crop_idx >= 0 and sim.crop == "":
                sim.crop = row[crop_idx].strip().strip('"')
            if variety_idx >= 0 and sim.variety == "":
                sim.variety = row[variety_idx].strip().strip('"')

            try:
                sow_doy = int(row[sowing_idx].strip())
            except (ValueError, IndexError):
                continue

            fung_doys = []
            for fi in fung_indices:
                if fi < len(row):
                    try:
                        d = int(row[fi].strip())
                        if d > 0:
                            fung_doys.append(d)
                    except ValueError:
                        pass
            fung_doys = sorted(set(fung_doys))

            year_cell = row[year_idx].strip().strip('"') if year_idx >= 0 else ""
            print(f'Parsed row: site={row_site}, year={year_cell}, variety={row_var}, sow_doy={sow_doy}, fung_doys={fung_doys}')
            
            if year_cell.lower() == "all":
                all_row = (sow_doy, fung_doys)
            else:
                try:
                    y = int(year_cell)
                    per_year[y] = (sow_doy, fung_doys)
                except ValueError:
                    pass

    # Apply the "All" row to every year in range. By default the final year is
    # excluded (half-open range); see ``all_row_includes_end_year``.
    all_end = end_year + 1 if all_row_includes_end_year else end_year
    if all_row is not None:
        for y in range(start_year, all_end):
            _apply_row(sim, y, all_row[0], all_row[1])

    # Override with per-year entries
    for y, (sow_doy, fung_doys) in per_year.items():
        _apply_row(sim, y, sow_doy, fung_doys)

    return sim


def _apply_row(sim: SimulationUnit, year: int, sow_doy: int, fung_doys: List[int]) -> None:
    """Populate sim_unit with sowing DOY and fungicide dates for a given year."""
    sim.year_sowing_doy[year] = sow_doy

    for f_doy in fung_doys:
        # Treatment in the *same* year as sowing unless DOY < sowing DOY (occurs next year)
        treat_year = year + 1 if f_doy < sow_doy else year
        treat_date = datetime(treat_year, 1, 1) + timedelta(days=f_doy - 1)
        sim.fungicide_treatment_schedule.add_treatment(treat_date)


def read_reference(
    ref_dir: str | Path,
    sowing_file: str | Path,
    site: str,
    variety: str,
    start_year: int,
    end_year: int,
    sim_unit: Optional[SimulationUnit] = None,
    disease: str = "thisDisease",
) -> SimulationUnit:
    """Read referenceData.csv and populate the SimulationUnit's reference_data.

    Args:
        ref_dir:    Directory containing referenceData.csv.
        sowing_file: Path to sowing.csv (loaded first if sim_unit is empty).
        site:       Site identifier.
        variety:    Variety identifier.
        start_year: First year.
        end_year:   Last year.
        sim_unit:   Existing SimulationUnit to populate; created if None.
        disease:    Disease column name to look for (e.g., "thisDisease", "stripe_rust").

    Returns:
        Updated SimulationUnit with reference_data populated.
    """
    if sim_unit is None or not sim_unit.year_sowing_doy:
        sim_unit = read_sowing(sowing_file, site, variety, start_year, end_year)

    sim_unit.site = site
    _rp = Path(ref_dir)
    path = _rp if _rp.is_file() else _rp / "referenceData.csv"

    if not path.exists():
        raise FileNotFoundError(f"referenceData.csv not found at '{path}'")
        # return sim_unit  # no reference data available

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        raw_headers = next(reader, [])
        header_map = {_norm(h): i for i, h in enumerate(raw_headers)}

        fint_col = _get_col(header_map, "fint", "f_int", "lightinterception", "lightint")
        agb_col = _get_col(header_map, "agb", "abovegroundbiomass", "biomass", "wtop")
        year_col = _get_col(header_map, "year", "yr")
        doy_col = _get_col(header_map, "doy", "dayofyear", "dy", "d")
        variety_col = _get_col(header_map, "variety", "cultivar", "cv")
        dis_col = _get_col(header_map, disease, f"{disease}sev", f"{disease}severity")

        yield_att_col = _get_col(header_map, "yieldattainable", "yieldunlimited", "yieldpotential",
                                 "yield", "wgrn", "grainyieldpotential")
        yield_act_col = _get_col(header_map, "yieldactual", "yielddiseased", "yieldact",
                                 "yieldlimited", "grainyieldlimited")
        
        if dis_col < 0:
            warnings.warn(
                f"[read_reference] Column for disease '{disease}' not found in {path}. "
                f"Available columns: {', '.join(raw_headers)}",
                stacklevel=2,
            )

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            if variety_col >= 0:
                row_var = row[variety_col].strip().strip('"').lower() if variety_col < len(row) else ""
                if row_var != variety.lower():
                    continue

            # Parse date from year + doy
            obs_date = None
            if year_col >= 0 and doy_col >= 0 and year_col < len(row) and doy_col < len(row):
                try:
                    y = int(row[year_col].strip())
                    doy = int(row[doy_col].strip())
                    if 1 <= doy <= 366:
                        obs_date = (datetime(y, 1, 1) + timedelta(days=doy - 1)).date()
                except ValueError:
                    pass
            if obs_date is None:
                obs_date = datetime.min.date()  # sentinel date for undated observations

            # FINT
            if fint_col >= 0:
                v = _pf(row[fint_col]) if fint_col < len(row) else None
                if v is not None:
                    sim_unit.reference_data.date_fint[obs_date] = v
            # AGB
            if agb_col >= 0:
                v = _pf(row[agb_col]) if agb_col < len(row) else None
                if v is not None:
                    sim_unit.reference_data.date_agb[obs_date] = v
            # Yield attainable
            if yield_att_col >= 0:
                v = _pf(row[yield_att_col]) if yield_att_col < len(row) else None
                if v is not None:
                    sim_unit.reference_data.date_yield_attainable[obs_date] = v
            # Yield actual
            if yield_act_col >= 0:
                v = _pf(row[yield_act_col]) if yield_act_col < len(row) else None
                if v is not None:
                    sim_unit.reference_data.date_yield_actual[obs_date] = v
            # Disease severity
            if dis_col >= 0:
                v = _pf(row[dis_col]) if dis_col < len(row) else None
                if v is not None:
                    sim_unit.reference_data.disease_date_disease_sev.setdefault(disease, {})[obs_date] = v
    return sim_unit


def read_crop_model_data(crop_model_file: str | Path, use_gdd: bool = False) -> CropModelData:
    """Read an external crop-model series and compute cycle progress.

    Loads the daily crop-model output (light interception, biomass, yield, and
    optionally thermal time) and segments it into growing cycles, computing the
    cycle-completion percentage for each day.

    Expected CSV columns (matched case-insensitively against common aliases):
    ``year``, ``doy``, ``fint``, ``agb``, ``yield``, and optionally ``gdd``.

    A new cycle is detected when the day-of-year steps backwards without a
    year boundary (a new sowing) or when yield resets from above to at or below
    the 100 kg ha⁻¹ harvest threshold.

    Cycle completion is computed as follows:

    - When ``use_gdd`` is ``True`` and thermal time is available, progress within
      a cycle is ``gdd[date] / max_gdd_in_cycle * 100``, which ties phenology to
      accumulated thermal time.
    - Otherwise, progress is interpolated linearly over calendar days between the
      first and last day of the cycle.

    Args:
        crop_model_file: Path to the directory containing ``cropModelData.csv``,
            or the path to the CSV file itself.
        use_gdd: Whether to derive cycle completion from thermal time rather than
            calendar days.

    Returns:
        Populated :class:`~simpest.models.fr_data.CropModelData`.
    """
    cmd = CropModelData()
    path = Path(crop_model_file)
    if path.is_dir():
        path = path / "cropModelData.csv"

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        raw_headers = next(reader, [])
        header_map = {_norm(h): i for i, h in enumerate(raw_headers)}

        fint_col = _get_col(header_map, "fint", "f_int", "lightinterception", "lightint")
        agb_col = _get_col(header_map, "agb", "abovegroundbiomass", "biomass", "wtop")
        yield_col = _get_col(header_map, "yield", "yieldattainable", "yieldunlimited", "yieldpotential", "wgrn", "grainyieldpotential")
        year_col = _get_col(header_map, "year", "yr")
        doy_col = _get_col(header_map, "doy", "dayofyear", "dy", "d")
        gdd_col = _get_col(header_map, "gdd", "growingdegreedays", "tsum", "thermaltime")

        date_order: List[datetime] = []

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            if year_col < 0 or doy_col < 0:
                continue

            try:
                y = int(row[year_col].strip())
                doy = int(row[doy_col].strip())
            except (ValueError, IndexError):
                continue

            try:
                dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
            except ValueError:
                continue

            d = dt.date()

            if fint_col >= 0 and fint_col < len(row):
                v = _pf(row[fint_col])
                if v is not None:
                    cmd.f_int[d] = v

            if agb_col >= 0 and agb_col < len(row):
                v = _pf(row[agb_col])
                if v is not None:
                    cmd.agb[d] = v

            if yield_col >= 0 and yield_col < len(row):
                v = _pf(row[yield_col])
                if v is not None:
                    cmd.yield_[d] = v

            if use_gdd and gdd_col >= 0 and gdd_col < len(row):
                v = _pf(row[gdd_col])
                if v is not None:
                    cmd.gdd[d] = v

            if d in cmd.f_int or d in cmd.agb or d in cmd.yield_:
                date_order.append(dt)

    if not date_order:
        return cmd

    # --- Cycle detection ---
    date_order_sorted = sorted(set(date_order))
    dates_d = [dt.date() for dt in date_order_sorted]

    cycles: List[Tuple[date, date]] = []
    cycle_start = dates_d[0]

    for i in range(1, len(dates_d)):
        prev = dates_d[i - 1]
        curr = dates_d[i]

        doy_backwards = curr.timetuple().tm_yday < prev.timetuple().tm_yday
        year_wrap = prev.month == 12 and curr.month == 1
        sowing_jump = doy_backwards and not year_wrap

        y_prev = cmd.yield_.get(prev, 0.0)
        y_curr = cmd.yield_.get(curr, 0.0)
        harvest_reset = (y_curr <= 100.0 and y_prev > 100.0)

        if sowing_jump or harvest_reset:
            cycles.append((cycle_start, prev))
            cycle_start = curr

    cycles.append((cycle_start, dates_d[-1]))

    # --- Compute cycle percentage ---
    for start, end in cycles:
        cycle_dates = [d for d in dates_d if start <= d <= end]
        if not cycle_dates:
            continue

        gdd_max = 0.0
        if use_gdd:
            gdd_values = [cmd.gdd.get(d, 0.0) for d in cycle_dates]
            gdd_max = max(gdd_values)

        if use_gdd and gdd_max > 0.0:
            # Thermal-time progression: scale by the cycle's maximum GDD
            for d in cycle_dates:
                gdd_d = cmd.gdd.get(d, 0.0)
                cmd.cycle_percentage[d] = min(100.0, gdd_d / gdd_max * 100.0)
        else:
            # Calendar-day interpolation between cycle start and end
            total_days = (end - start).days
            if total_days <= 0:
                continue
            for d in cycle_dates:
                frac = (d - start).days / total_days
                cmd.cycle_percentage[d] = frac * 100.0

    return cmd
