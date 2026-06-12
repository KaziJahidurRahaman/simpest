"""Weather readers that build an hourly forcing series for the model.

The disease sub-model operates on an hourly time step, so these readers ingest
either daily or hourly weather CSV files and return a complete hourly series.
When the input is daily, a physically based diurnal synthesis fills in the 24
hourly records for each day.

Key behaviour:

- **Header parsing.** Column names are matched case-insensitively against a set
  of common aliases, so files from different sources are accepted without
  renaming.
- **Date columns.** A single ``Date``/``Datetime`` column or separate
  ``Year``/``Month``/``Day`` (plus ``Hour`` for hourly files) are both accepted.
- **Radiation.** Measured radiation is used when present; otherwise daily global
  solar radiation is estimated with the Hargreaves–Samani relationship from the
  daily temperature range and distributed over the day by clear-sky
  (extraterrestrial) fractions.
- **Humidity.** Relative humidity is reconstructed from daily ``RHx``/``RHn``
  extrema with a cosine diurnal curve, or estimated from a dew-point relation
  when extrema are absent.
"""

from __future__ import annotations

import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .fr_data import InputsHourly, InputsDaily


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_SOLAR_CONSTANT = 4.921      # MJ m⁻² h⁻¹
_DEG_TO_RAD = math.pi / 180.0
_KRS = 0.19                  # Hargreaves coefficient (coastal; 0.16 for inland)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_daily(
    file: str | Path,
    start_year: int,
    end_year: int,
    latitude: float = 0.0,
) -> Dict[datetime, InputsHourly]:
    """Read a daily weather CSV and return synthesized hourly records.

    Supports year/month/day columns OR a single date/datetime column.
    Radiation is estimated when absent (Hargreaves-Samani) if latitude is known.

    Args:
        file:       Path to the daily weather CSV.
        start_year: First year to include (inclusive).
        end_year:   Last year to include (inclusive).
        latitude:   Fallback latitude in decimal degrees when not in CSV.

    Returns:
        Dict keyed by ``datetime(year, month, day, hour)`` → ``InputsHourly``.
    """
    result: Dict[datetime, InputsHourly] = {}
    path = Path(file)

    with path.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample.split("\n")[0] else ","

        reader = csv.reader(fh, delimiter=delimiter)
        headers = [_clean(h) for h in next(reader)]
        col = {name: i for i, name in enumerate(headers)}

        # Date columns
        date_idx = _idx(col, ["date", "datetime", "timestamp"], optional=True)
        year_idx = _idx(col, ["year"], optional=True)
        month_idx = _idx(col, ["month"], optional=True)
        day_idx = _idx(col, ["day"], optional=True)

        has_ymd = (year_idx >= 0 and month_idx >= 0 and day_idx >= 0)
        has_date = date_idx >= 0


        if not has_ymd and not has_date:
            raise ValueError(
                "Weather file must contain (year, month, day) OR a single date/datetime column."
            )

        # Meteorological columns
        rad_idx = _idx(col, ["rad", "radiation", "solar", "solarrad", "srad"], optional=True)
        lat_idx = _idx(col, ["lat", "latitude", "sitelat", "phi"], optional=True)
        tmax_idx = _idx(col, ["tmax", "t2mmax", "maxtemp", "tx"])
        tmin_idx = _idx(col, ["tmin", "t2mmin", "mintemp", "tn"])
        prec_idx = _idx(col, ["prec", "precip", "rain", "rainfall", "precipitation", "p"], optional=True)
        rhx_idx = _idx(col, ["rhmax", "humiditymax", "relativehumiditymax", "hummax", "rhx"], optional=True)
        rhn_idx = _idx(col, ["rhmin", "humiditymin", "relativehumiditymin", "hummin", "rhn"], optional=True)

        if rad_idx < 0 and lat_idx < 0 and latitude == 0.0:
            raise ValueError(
                "Weather file must contain a radiation column OR a latitude column "
                "(or pass latitude= to read_daily)."
            )

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            # --- Date ---
            try:
                if has_ymd:
                    yr = int(row[year_idx])
                    mo = int(row[month_idx])
                    dy = int(row[day_idx])
                    day_date = date(yr, mo, dy)
                else:
                    raw_date = row[date_idx].strip().strip('"')
                    day_date = datetime.fromisoformat(raw_date).date()
            except (ValueError, IndexError):
                continue

            if not (start_year <= day_date.year <= end_year):
                continue

            # --- Temperatures ---
            try:
                tmax = _pf(row, tmax_idx)
                tmin = _pf(row, tmin_idx)
            except (ValueError, IndexError):
                continue

            # --- Radiation ---
            rad = _pf(row, rad_idx) if rad_idx >= 0 else float("nan")
            rad_missing = not math.isfinite(rad) or rad <= 0.0

            # --- Latitude (for radiation estimation) ---
            lat = _pf(row, lat_idx) if lat_idx >= 0 else latitude
        
            if not math.isfinite(lat):
                lat = latitude

            if rad_missing:
                if lat == 0.0:
                    continue  # cannot estimate without latitude
                rd = _day_length(day_date, lat, tmax, tmin)
                rad = rd["gsr"]
                if not math.isfinite(rad) or rad <= 0.0:
                    continue

            # --- Build InputsDaily ---
            prec = _pf(row, prec_idx) if prec_idx >= 0 else 0.0
            if not math.isfinite(prec):
                prec = 0.0

            id_ = InputsDaily(
                date=datetime(day_date.year, day_date.month, day_date.day),
                tmax=tmax,
                tmin=tmin,
                rad=rad,
                precipitation=prec,
                latitude=lat,
            )
            id_.dew_point = _dew_point(id_.tmax, id_.tmin)

            if rhx_idx >= 0:
                v = _pf(row, rhx_idx)
                if math.isfinite(v):
                    id_.rhx = v
            if rhn_idx >= 0:
                v = _pf(row, rhn_idx)
                if math.isfinite(v):
                    id_.rhn = v
            
            # Calculate leaf wetness from RH and precipitation
            # High humidity (>80%) or significant precipitation (>0.5mm) indicate wet leaves
            avg_rh = (id_.rhx + id_.rhn) / 2.0 if id_.rhx > 0 else 0.0
            if prec >= 0.5 or avg_rh >= 80.0:
                # Estimate wetness hours: scale from 60% RH (minimal) to 100% RH (24 hours)
                if avg_rh >= 80.0:
                    id_.leaf_wetness = min(24.0, (avg_rh - 80.0) * 1.2)  # 80% -> 0h, 100% -> 24h
                else:
                    id_.leaf_wetness = 0.0
                # Add hours if precipitation present
                if prec >= 0.5:
                    id_.leaf_wetness = min(24.0, id_.leaf_wetness + prec * 2.0)  # Add ~2h per mm rainfall

            # --- Synthesize 24 hourly records ---
            hourly = _estimate_hourly(id_, day_date, lat)
            result.update(hourly)

    return result


def read_hourly(
    file: str | Path,
    start_year: int,
    end_year: int,
    site: str = "",
    latitude: float = 0.0,
) -> Dict[datetime, InputsHourly]:
    """Read an hourly weather CSV and return the hourly records.

    Missing hourly radiation is filled per-day using the Hargreaves estimate
    distributed via clear-sky (ETR) fractions.

    Args:
        file:       Path to the hourly weather CSV.
        start_year: First year to include (inclusive).
        end_year:   Last year to include (inclusive).
        site:       Optional site filter (column 'site' or 'station').
        latitude:   Fallback latitude when not in the file.

    Returns:
        Dict keyed by ``datetime(year, month, day, hour)`` → ``InputsHourly``.
    """
    result: Dict[datetime, InputsHourly] = {}
    daily_buckets: Dict[date, List[InputsHourly]] = {}

    path = Path(file)

    with path.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample.split("\n")[0] else ","
        reader = csv.reader(fh, delimiter=delimiter)
        headers = [_clean(h) for h in next(reader)]
        col = {name: i for i, name in enumerate(headers)}

        year_idx = _idx(col, ["year"], optional=True)
        month_idx = _idx(col, ["month", "mo"], optional=True)
        day_idx = _idx(col, ["day", "dd", "dy"], optional=True)
        hour_idx = _idx(col, ["hour", "hr", "h"], optional=True)
        date_idx = _idx(col, ["date", "datetime", "timestamp"], optional=True)

        has_ymdh = (year_idx >= 0 and month_idx >= 0 and day_idx >= 0 and hour_idx >= 0)
        has_dateh = (date_idx >= 0 and hour_idx >= 0)

        if not has_ymdh and not has_dateh:
            raise ValueError(
                "Hourly file must contain (year,month,day,hour) OR (date,hour)."
            )

        tmax_idx = _idx(col, ["tmax", "t2mmax", "maxtemp"], optional=True)
        tmin_idx = _idx(col, ["tmin", "t2mmin", "mintemp"], optional=True)
        temp_idx = _idx(col, ["temp", "temperature", "t2m"], optional=True)
        prec_idx = _idx(col, ["prec", "precip", "precipitation", "prectotcorr", "rain", "rainfall"], optional=True)
        rh_idx = _idx(col, ["rh", "humidity", "relhumidity", "relativehumidity"], optional=True)
        rad_idx = _idx(col, ["rad", "radiation", "solar", "solarrad"], optional=True)
        lat_idx = _idx(col, ["latitude", "lat"], optional=True)

        if rad_idx < 0 and lat_idx < 0 and latitude == 0.0:
            raise ValueError(
                "Hourly weather file must contain a radiation column OR a latitude column."
            )

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            # --- Timestamp ---
            try:
                if has_ymdh:
                    yr = int(row[year_idx])
                    mo = int(row[month_idx])
                    dy = int(row[day_idx])
                    hr = int(row[hour_idx])
                    ts = datetime(yr, mo, dy, hr)
                else:
                    raw = row[date_idx].strip().strip('"')
                    ts = datetime.fromisoformat(raw)
                    ts = ts.replace(minute=0, second=0, microsecond=0)
                    hr = int(row[hour_idx])
                    ts = ts.replace(hour=hr)
            except (ValueError, IndexError):
                continue

            if not (start_year <= ts.year <= end_year):
                continue

            gw = InputsHourly(date=ts)

            # Temperature
            if temp_idx >= 0:
                v = _pf(row, temp_idx)
                if math.isfinite(v):
                    gw.air_temperature = v
            elif tmax_idx >= 0 and tmin_idx >= 0:
                tx = _pf(row, tmax_idx)
                tn = _pf(row, tmin_idx)
                if math.isfinite(tx) and math.isfinite(tn):
                    gw.air_temperature = 0.5 * (tx + tn)

            # Precipitation
            if prec_idx >= 0:
                v = _pf(row, prec_idx)
                if math.isfinite(v) and v >= 0:
                    gw.precipitation = v

            # Relative humidity
            if rh_idx >= 0:
                v = _pf(row, rh_idx)
                if math.isfinite(v):
                    gw.relative_humidity = max(0.0, min(100.0, v))
            else:
                t = gw.air_temperature
                dp = _dew_point(t, t)
                es = 0.61121 * math.exp((17.502 * t) / (240.97 + t))
                ea = 0.61121 * math.exp((17.502 * dp) / (240.97 + dp))
                gw.relative_humidity = max(0.0, min(100.0, ea / es * 100.0)) if es > 0 else 0.0

            # Radiation
            if rad_idx >= 0:
                v = _pf(row, rad_idx)
                if math.isfinite(v) and v >= 0:
                    gw.rad = v

            # Latitude
            if lat_idx >= 0:
                v = _pf(row, lat_idx)
                if math.isfinite(v):
                    gw.latitude = v
            elif latitude != 0.0:
                gw.latitude = latitude

            # Leaf wetness
            gw.leaf_wetness = 1.0 if (gw.relative_humidity > 90.0 or gw.precipitation >= 0.2) else 0.0

            day_key = ts.date()
            daily_buckets.setdefault(day_key, []).append(gw)

    # --- Post-process: fill missing hourly radiation per day ---
    for day_key, records in daily_buckets.items():
        if not records:
            continue

        lat = records[0].latitude if records[0].latitude else latitude
        temps = [r.air_temperature for r in records]
        tmax_d = max(temps)
        tmin_d = min(temps)

        all_rad_zero = all(r.rad <= 0.0 for r in records)

        rd = None
        if all_rad_zero and lat:
            rd = _day_length(day_key, lat, tmax_d, tmin_d)

        for rec in records:
            hr = rec.date.hour
            if rec.rad <= 0.0 and rd is not None:
                rec.rad = rd["gsr_hourly"][hr]
            rec.leaf_wetness = 1.0 if (rec.relative_humidity > 90.0 or rec.precipitation >= 0.2) else 0.0
            result[rec.date] = rec

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean(name: str) -> str:
    """Normalize a column header: lowercase, remove spaces/underscores/quotes."""
    return name.strip().strip('"').lower().replace(" ", "").replace("_", "")


def _idx(col: dict, aliases: list[str], optional: bool = False) -> int:
    """Return column index for the first matching alias, or -1 if optional and missing."""
    for a in aliases:
        key = _clean(a)
        if key in col:
            return col[key]
    if not optional:
        raise ValueError(f"Missing required weather column: {aliases}")
    return -1


def _pf(row: list[str], idx: int) -> float:
    """Parse a float from a CSV row at index idx."""
    try:
        return float(row[idx].strip().strip('"'))
    except (ValueError, IndexError):
        return float("nan")


def _dew_point(tmax: float, tmin: float) -> float:
    """Empirical dew-point estimate from daily extremes.

    Uses the regression

    $$
    T_{dew} = 0.38\\,T_{max} - 0.018\\,T_{max}^2 + 1.4\\,T_{min} - 5
    $$

    Args:
        tmax (float): Daily maximum temperature (°C).
        tmin (float): Daily minimum temperature (°C).

    Returns:
        float: Estimated dew-point temperature (°C).
    """
    return 0.38 * tmax - 0.018 * (tmax ** 2) + 1.4 * tmin - 5.0


def _estimate_hourly_rh(rh_min: float, rh_max: float, hour: int) -> float:
    """Cosine-interpolated RH between daily min (≈14:00) and max."""
    radians = math.pi * (hour - 14) / 12.0
    rh = rh_min + (rh_max - rh_min) * 0.5 * (1 + math.cos(radians))
    return max(0.0, min(100.0, rh))


def _day_length(day_date: date, lat_deg: float, tmax: float, tmin: float) -> dict:
    """Compute solar geometry and estimate daily GSR via Hargreaves-Samani.

    Returns a dict with:
        gsr          – daily global solar radiation (MJ m⁻² d⁻¹)
        etr          – daily extraterrestrial radiation (MJ m⁻² d⁻¹)
        day_length_h – photoperiod (hours)
        sunrise      – approximate hour of sunrise
        sunset       – approximate hour of sunset
        etr_hourly   – list[24] hourly ETR (MJ m⁻² h⁻¹)
        gsr_hourly   – list[24] distributed hourly GSR (MJ m⁻² h⁻¹)
    """
    doy = day_date.timetuple().tm_yday
    lat_rad = lat_deg * _DEG_TO_RAD

    # Solar geometry
    inv_earth_sun = 1.0 + 0.0334 * math.cos(0.01721 * doy - 0.0552)
    solar_decl = 0.4093 * math.sin((6.284 / 365.0) * (284 + doy))
    sin_dec = math.sin(solar_decl)
    cos_dec = math.cos(solar_decl)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Sunset hour angle
    ws_arg = max(-1.0, min(1.0, -math.tan(solar_decl) * math.tan(lat_rad)))
    ws = math.acos(ws_arg)

    # Hourly ETR
    etr_hourly = [0.0] * 24
    etr_sum = 0.0
    for h in range(24):
        hour_angle_deg = 15.0 * (h - 12.0)
        sin_elev = max(0.0, sin_dec * sin_lat + cos_dec * cos_lat * math.cos(hour_angle_deg * _DEG_TO_RAD))
        hly_etr = _SOLAR_CONSTANT * inv_earth_sun * sin_elev
        etr_hourly[h] = hly_etr
        etr_sum += hly_etr

    # Analytical daily ETR (stable for |lat| < 65°)
    if -65 < lat_deg < 65:
        etr = (24.0 / math.pi) * _SOLAR_CONSTANT * inv_earth_sun * (
            ws * (sin_dec * sin_lat) + cos_dec * cos_lat * math.sin(ws)
        )
        day_length_h = (24.0 / math.pi) * ws
    else:
        etr = etr_sum
        day_length_h = sum(1 for v in etr_hourly if v > 0)

    # Hargreaves-Samani daily GSR
    if tmax < tmin:
        tmax, tmin = tmin, tmax
    td = max(0.0, tmax - tmin)
    gsr = _KRS * math.sqrt(td) * etr
    gsr = max(0.0, min(gsr, etr))
    gsr = round(gsr, 2)

    # Distribute GSR hourly by clear-sky fraction
    gsr_hourly = [0.0] * 24
    for h in range(24):
        frac = etr_hourly[h] / etr if etr > 0.0 else 0.0
        gsr_hourly[h] = frac * gsr

    sunrise = 12.0 - day_length_h / 2.0
    sunset = 12.0 + day_length_h / 2.0

    return {
        "gsr": gsr,
        "etr": etr,
        "day_length_h": day_length_h,
        "sunrise": sunrise,
        "sunset": sunset,
        "etr_hourly": etr_hourly,
        "gsr_hourly": gsr_hourly,
    }


def _estimate_hourly(
    input_daily: InputsDaily,
    day_date: date,
    lat_deg: float = 0.0,
) -> Dict[datetime, InputsHourly]:
    """Synthesize 24 hourly InputsHourly records from a daily InputsDaily.

    The synthesis applies the following diurnal models:

    - **Temperature:** cosine diurnal cycle peaking at 15:00.
    - **Relative humidity:** cosine interpolation between the daily ``RHx`` and
      ``RHn`` extrema, or a dew-point based estimate when extrema are absent.
    - **Precipitation:** split uniformly across the 24 hours.
    - **Radiation:** split uniformly across the 24 hours (``rad_daily / 24``),
      which preserves the measured or estimated daily total. The intra-day shape
      does not affect downstream results because the runner re-aggregates hourly
      radiation into the daily total before it is consumed.

    Args:
        input_daily (InputsDaily): Daily record to expand.
        day_date (date): Calendar date of the day being synthesised.
        lat_deg (float): Site latitude in decimal degrees.

    Returns:
        Dict[datetime, InputsHourly]: The 24 synthesised hourly records keyed by
        timestamp.
    """
    records: Dict[datetime, InputsHourly] = {}
    tmax = input_daily.tmax
    tmin = input_daily.tmin
    avg_t = (tmax + tmin) / 2.0
    daily_range = tmax - tmin
    dew = input_daily.dew_point
    rain = input_daily.precipitation
    rad_daily = input_daily.rad
    rhx = input_daily.rhx    # 0.0 means absent
    rhn = input_daily.rhn

    for hr in range(24):
        ts = datetime(day_date.year, day_date.month, day_date.day, hr)

        # Temperature: cosine wave, peak ≈ 15:00
        hourly_t = avg_t + (daily_range / 2.0) * math.cos(0.2618 * (hr - 15))

        # RH
        if rhx > 0.0:
            rh_h = _estimate_hourly_rh(rhn, rhx, hr)
        else:
            es = 0.61121 * math.exp((17.502 * hourly_t) / (240.97 + hourly_t))
            ea = 0.61121 * math.exp((17.502 * dew) / (240.97 + dew))
            rh_h = min(100.0, ea / es * 100.0) if es > 0.0 else 0.0

        # Radiation: uniform split across the day, preserving the daily total
        rad_h = rad_daily / 24.0

        # Precipitation (uniform split)
        prec_h = rain / 24.0

        # Leaf wetness
        lw = 1.0 if (prec_h >= 0.2 or rh_h >= 90.0) else 0.0

        gw = InputsHourly(
            date=ts,
            air_temperature=hourly_t,
            precipitation=prec_h,
            relative_humidity=rh_h,
            leaf_wetness=lw,
            rad=rad_h,
            latitude=lat_deg,
        )
        records[ts] = gw

    return records
