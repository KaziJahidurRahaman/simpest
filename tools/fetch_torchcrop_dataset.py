"""Build a torchcrop-format site/soil/weather dataset for a new location.

Reproduces the directory layout used by the Brandenburg example dataset
shipped with torchcrop (``docs/examples/torchcrop-docs/examples/data/brandenburg``)::

    <out-dir>/
        site/site.csv        one row: location, NUTS_ID, NUTS_NAME, STATE_NAME,
                              LATITUDE, ALTITUDE, IDPL, CO2
        soil/soil.csv         one row: location, SMDRY, SMW, SMFC, SMO, CRAIRC,
                              SMI, SMLOWI, RDMSO, RUNFR, CFEV, KSUB, NMINS,
                              PMINS, KMINS, RTNMINS, RTPMINS, RTKMINS
        weather/<id>.csv      daily: Date, TempMean, TempMin, TempMax,
                              Radiation, Precipitation, VapPressure, Windspeed

Only ``weather`` is a genuine per-day fetch. ``site``/``soil`` are single
static rows assembled from a handful of lookups plus the generic LINTUL5/
SIMPLACE soil defaults that the Brandenburg reference set itself uses for
every location (see ``GENERIC_SOIL_DEFAULTS`` below) -- treat the derived
water-retention numbers (SMDRY/SMW/SMFC/SMO) as estimates to sanity-check
before a real calibration run, not as reference-quality values.

Data sources (all free, no API key required):
    weather   Open-Meteo historical archive (ERA5 / ERA5-Land reanalysis),
              https://archive-api.open-meteo.com -- covers 1940-present,
              so it (unlike NASA POWER, which starts in 1981) can serve the
              full 1970-2000 range in one consistent source.
    elevation Open-Meteo elevation API, https://api.open-meteo.com/v1/elevation
    soil texture SoilGrids v2.0 REST API, https://rest.isric.org/soilgrids/v2.0
    county/state (optional, cosmetic) OpenStreetMap Nominatim reverse geocoding

Usage:
    .venv/Scripts/python.exe tools/fetch_torchcrop_dataset.py \\
        --lat 40.4698 --lon -86.9927 \\
        --start-year 1970 --end-year 2000 \\
        --location-name west_lafayette_in
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "torchcrop-dataset-builder/1.0 (research use; contact via repo issue tracker)"

# NOAA GML Mauna Loa annual mean atmospheric CO2 (ppm), 1970-2000. Static
# reference table (not live-fetched) -- copied verbatim from
# https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt
# (fetched 2026-08-11). NOAA revises earlier years occasionally as they
# recalibrate reference gas mixtures, so re-pull that file if exactness
# against the current published series matters.
MAUNA_LOA_ANNUAL_CO2 = {
    1970: 325.68, 1971: 326.32, 1972: 327.46, 1973: 329.68, 1974: 330.19,
    1975: 331.13, 1976: 332.03, 1977: 333.84, 1978: 335.41, 1979: 336.84,
    1980: 338.76, 1981: 340.12, 1982: 341.48, 1983: 343.15, 1984: 344.87,
    1985: 346.35, 1986: 347.61, 1987: 349.31, 1988: 351.69, 1989: 353.20,
    1990: 354.45, 1991: 355.70, 1992: 356.54, 1993: 357.21, 1994: 358.96,
    1995: 360.97, 1996: 362.74, 1997: 363.88, 1998: 366.84, 1999: 368.54,
    2000: 369.71,
}

# Soil parameters that the Brandenburg reference set holds constant across all
# 18 of its locations -- LINTUL5/SIMPLACE model defaults, not per-location
# fetched or texture-derived values, so reusing them elsewhere is no worse
# than what the reference set itself does. SMI/SMLOWI there also equal SMFC
# exactly (root zone initialised at field capacity), and SMDRY is always
# exactly half of SMW, so both are derived rather than looked up.
#
# KSUB (max percolation to deeper layers) is deliberately NOT here: unlike
# these, it's a proxy for saturated hydraulic conductivity, which is strongly
# texture-dependent -- Brandenburg's sandy soils vs. a fine-textured site
# elsewhere would give very different real values, so it's computed per-site
# from the fetched texture instead (see `saxton_rawls_2006`).
GENERIC_SOIL_DEFAULTS = {
    "CRAIRC": 0.07,
    "RDMSO": 2.0,
    "RUNFR": 0.0,
    "CFEV": 2.0,
    "NMINS": 3.0,
    "PMINS": 3.0,
    "KMINS": 3.0,
    "RTNMINS": 0.025,
    "RTPMINS": 0.025,
    "RTKMINS": 0.025,
}

FAO56_10M_TO_2M = 4.87 / np.log(67.8 * 10 - 5.42)  # ~0.7484, wind height correction


def http_get_json(url: str, params: dict, timeout: float = 90.0, retries: int = 4, backoff: float = 5.0) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            print(f"    request failed ({e}); retrying in {backoff:.0f}s [{attempt}/{retries}]")
            time.sleep(backoff)
    raise RuntimeError(f"failed to fetch {full_url}: {last_err}")


def fetch_elevation(lat: float, lon: float) -> float:
    data = http_get_json(ELEVATION_URL, {"latitude": lat, "longitude": lon})
    return float(data["elevation"][0])


def fetch_soil_texture(lat: float, lon: float) -> dict:
    """Depth-weighted (0-30cm) clay/sand/silt/organic-matter percent from SoilGrids."""
    params = [
        ("lon", lon), ("lat", lat),
        ("property", "clay"), ("property", "sand"),
        ("property", "silt"), ("property", "soc"),
        ("depth", "0-5cm"), ("depth", "5-15cm"), ("depth", "15-30cm"),
        ("value", "mean"),
    ]
    query = urllib.parse.urlencode(params)
    data = None
    full_url = f"{SOILGRIDS_URL}?{query}"
    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=90.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            print(f"    SoilGrids request failed ({e}); retrying in 5s [{attempt}/4]")
            time.sleep(5.0)
    if data is None:
        raise RuntimeError(f"failed to fetch SoilGrids texture: {last_err}")

    thickness = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
    out = {}
    for layer in data["properties"]["layers"]:
        d_factor = layer["unit_measure"]["d_factor"]
        target_units = layer["unit_measure"]["target_units"]
        total_w, total = 0.0, 0.0
        for depth in layer["depths"]:
            mean = depth["values"]["mean"]
            if mean is None:
                continue
            w = thickness[depth["label"]]
            total += (mean / d_factor) * w
            total_w += w
        if total_w == 0:
            raise ValueError(
                f"SoilGrids returned no data for property {layer['name']!r} at "
                f"({lat}, {lon}) -- location may be open water or outside coverage."
            )
        value = total / total_w
        if target_units == "g/kg":  # SOC comes back as g/kg, not %; 1% = 10 g/kg
            value /= 10.0
        elif target_units != "%":
            raise ValueError(f"unexpected SoilGrids unit {target_units!r} for {layer['name']!r}")
        out[layer["name"]] = value
    return {
        "clay_pct": out["clay"],
        "sand_pct": out["sand"],
        "silt_pct": out["silt"],
        "om_pct": out["soc"] * 1.724,  # van Bemmelen factor, SOC% -> OM%
    }


def reverse_geocode(lat: float, lon: float) -> dict:
    """Best-effort county/state lookup for the cosmetic site.csv columns."""
    try:
        data = http_get_json(
            NOMINATIM_URL,
            {"lat": lat, "lon": lon, "format": "json", "zoom": 8},
            retries=1,
        )
        address = data.get("address", {})
        county = address.get("county") or address.get("state_district") or "NA"
        state = address.get("state") or "NA"
        iso = address.get("ISO3166-2-lvl4", "NA")
        return {"nuts_id": iso, "nuts_name": county, "state_name": state}
    except Exception as e:
        print(f"    reverse geocoding failed ({e}); leaving NUTS/state fields as 'NA'")
        return {"nuts_id": "NA", "nuts_name": "NA", "state_name": "NA"}


def saxton_rawls_2006(sand_pct: float, clay_pct: float, om_pct: float) -> dict:
    """Saxton & Rawls (2006) soil-water-characteristic pedotransfer function.

    Coefficients cross-checked against the USDA WEPP Fortran implementation
    (``saxpar.for``, https://infosys.ars.usda.gov/svn/code/weps1/branches/
    weps.src.wepp/src/lib_wepp/saxpar.for) -- sand/clay are fractions
    [0, 1], organic matter is percent, matching the physically sane range
    of the outputs (plugging sand/clay in as 0-100 percent instead, as that
    file's own inline units comment claims, produces water contents > 1
    m3 m-3, which is impossible; fractions do not).

    Returns volumetric water content [m3 m-3] at wilting point (1500 kPa),
    field capacity (33 kPa), saturation, and saturated hydraulic
    conductivity ``ksat_mm_day`` [mm d-1] (the same Ksat equation from that
    file, converted from its native mm h-1).
    """
    s = sand_pct / 100.0
    c = clay_pct / 100.0
    om = om_pct

    theta_1500t = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * (s * om) - 0.013 * (c * om) + 0.068 * (s * c) + 0.031
    wp = theta_1500t + (0.14 * theta_1500t - 0.02)

    theta_33t = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * (s * om) - 0.027 * (c * om) + 0.452 * (s * c) + 0.299
    fc = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

    theta_s33t = 0.278 * s + 0.034 * c + 0.022 * om - 0.018 * (s * om) - 0.027 * (c * om) - 0.584 * (s * c) + 0.078
    theta_s33 = theta_s33t + (0.636 * theta_s33t - 0.107)

    sat = fc + theta_s33 - 0.097 * s + 0.043

    if not (0.0 < wp < fc < sat < 1.0):
        raise ValueError(
            f"Saxton-Rawls PTF produced non-physical water contents "
            f"(wp={wp:.3f}, fc={fc:.3f}, sat={sat:.3f}) for sand={sand_pct:.1f}%, "
            f"clay={clay_pct:.1f}%, OM={om_pct:.1f}% -- inputs are likely out of range."
        )

    b_slope = (np.log(1500) - np.log(33)) / (np.log(fc) - np.log(wp))
    ksat_mm_h = 1930.0 * (sat - fc) ** (3.0 - 1.0 / b_slope)
    ksat_mm_day = ksat_mm_h * 24.0

    return {"wp": wp, "fc": fc, "sat": sat, "ksat_mm_day": ksat_mm_day}


def build_site_row(location_id: int, lat: float, lon: float, idpl: int, co2: float) -> dict:
    elevation = fetch_elevation(lat, lon)
    geo = reverse_geocode(lat, lon)
    return {
        "location": location_id,
        "NUTS_ID": geo["nuts_id"],
        "NUTS_NAME": geo["nuts_name"],
        "STATE_NAME": geo["state_name"],
        "LATITUDE": round(lat, 4),
        "ALTITUDE": round(elevation, 1),
        "IDPL": idpl,
        "CO2": round(co2, 1),
    }


def build_soil_row(location_id: int, lat: float, lon: float) -> dict:
    texture = fetch_soil_texture(lat, lon)
    wrc = saxton_rawls_2006(texture["sand_pct"], texture["clay_pct"], texture["om_pct"])
    smw = wrc["wp"]
    smfc = wrc["fc"]
    smo = wrc["sat"]
    ksub = wrc["ksat_mm_day"]
    row = {
        "location": location_id,
        "SMDRY": round(0.5 * smw, 6),
        "SMW": round(smw, 6),
        "SMFC": round(smfc, 6),
        "SMO": round(smo, 6),
        "SMI": round(smfc, 6),
        "SMLOWI": round(smfc, 6),
        "KSUB": round(ksub, 1),
        **GENERIC_SOIL_DEFAULTS,
    }
    print(
        f"    texture: clay={texture['clay_pct']:.1f}% sand={texture['sand_pct']:.1f}% "
        f"silt={texture['silt_pct']:.1f}% OM={texture['om_pct']:.1f}%  ->  "
        f"WP={smw:.3f} FC={smfc:.3f} SAT={smo:.3f} m3/m3, KSUB={ksub:.1f} mm/day (Saxton-Rawls, "
        f"replaces the Brandenburg flat default)"
    )
    return row


def fetch_daily_block(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    data = http_get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                     "precipitation_sum,shortwave_radiation_sum",
            "timezone": "auto",
        },
    )
    d = data["daily"]
    return pd.DataFrame(
        {
            "Date": d["time"],
            "TempMean": d["temperature_2m_mean"],
            "TempMin": d["temperature_2m_min"],
            "TempMax": d["temperature_2m_max"],
            "Radiation": [v * 1000.0 if v is not None else None for v in d["shortwave_radiation_sum"]],  # MJ -> kJ
            "Precipitation": d["precipitation_sum"],
        }
    )


def fetch_hourly_derived_block(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily-mean actual vapour pressure [kPa] and 2m wind speed [m/s].

    Both are derived from hourly fields since Open-Meteo's daily archive
    endpoint doesn't expose a daily mean humidity or wind speed directly.
    """
    data = http_get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "auto",
        },
    )
    h = data["hourly"]
    df = pd.DataFrame(
        {
            "datetime": h["time"],
            "temperature_2m": h["temperature_2m"],
            "relative_humidity_2m": h["relative_humidity_2m"],
            "wind_speed_10m": h["wind_speed_10m"],
        }
    )
    df["Date"] = df["datetime"].str.slice(0, 10)

    t = df["temperature_2m"].astype(float)
    rh = df["relative_humidity_2m"].astype(float)
    sat_vp = 0.6108 * np.exp(17.27 * t / (t + 237.3))  # Tetens/FAO-56, kPa
    df["vap_pressure"] = sat_vp * rh / 100.0
    df["wind_2m"] = df["wind_speed_10m"].astype(float) * FAO56_10M_TO_2M

    daily = df.groupby("Date", as_index=False).agg(
        VapPressure=("vap_pressure", "mean"),
        Windspeed=("wind_2m", "mean"),
    )
    return daily


def year_chunks(start_year: int, end_year: int, chunk_years: int):
    y = start_year
    while y <= end_year:
        y_end = min(y + chunk_years - 1, end_year)
        yield y, y_end
        y = y_end + 1


def fetch_weather(lat: float, lon: float, start_year: int, end_year: int, chunk_years: int, sleep_s: float) -> pd.DataFrame:
    frames = []
    chunks = list(year_chunks(start_year, end_year, chunk_years))
    for i, (y0, y1) in enumerate(chunks, 1):
        start_date, end_date = f"{y0}-01-01", f"{y1}-12-31"
        print(f"  [{i}/{len(chunks)}] fetching {start_date}..{end_date}")
        daily = fetch_daily_block(lat, lon, start_date, end_date)
        time.sleep(sleep_s)
        derived = fetch_hourly_derived_block(lat, lon, start_date, end_date)
        time.sleep(sleep_s)
        merged = daily.merge(derived, on="Date", how="left")
        frames.append(merged)

    weather = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

    n_missing = weather.isna().any(axis=1).sum()
    if n_missing:
        print(f"  WARNING: {n_missing} day(s) have a missing value; filling by linear interpolation")
        weather = weather.set_index("Date")
        weather = weather.interpolate(limit=3, limit_direction="both")
        weather = weather.reset_index()
        still_missing = weather.isna().any(axis=1).sum()
        if still_missing:
            raise RuntimeError(
                f"{still_missing} day(s) still have missing weather values after "
                "interpolation -- gap is too large to fill automatically, inspect the source data."
            )

    weather["TempMean"] = weather["TempMean"].round(1)
    weather["TempMin"] = weather["TempMin"].round(1)
    weather["TempMax"] = weather["TempMax"].round(1)
    weather["Radiation"] = weather["Radiation"].round(0).astype(int)
    weather["Precipitation"] = weather["Precipitation"].round(1)
    weather["VapPressure"] = weather["VapPressure"].round(2)
    weather["Windspeed"] = weather["Windspeed"].round(1)
    return weather[["Date", "TempMean", "TempMin", "TempMax", "Radiation", "Precipitation", "VapPressure", "Windspeed"]]


def write_csv(rows, columns, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, default=40.4698, help="Site latitude [deg N]")
    parser.add_argument("--lon", type=float, default=-86.9927, help="Site longitude [deg E, negative = W]")
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=2000)
    parser.add_argument("--location-id", type=int, default=0, help="Numeric id used for weather/<id>.csv and the 'location' column")
    parser.add_argument("--location-name", type=str, default="west_lafayette_in", help="Output subfolder name")
    parser.add_argument("--idpl", type=int, default=270, help="Sowing day-of-year (270 = late Sept, matches the Brandenburg winter-wheat reference)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output torchcrop/ dir (default: docs/examples/data/<location-name>/torchcrop)")
    parser.add_argument("--chunk-years", type=int, default=5, help="Years per API request block")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API requests")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("docs/examples/data") / args.location_name / "torchcrop"
    print(f"Output directory: {out_dir}")

    co2_years = [y for y in range(args.start_year, args.end_year + 1) if y in MAUNA_LOA_ANNUAL_CO2]
    if co2_years:
        co2 = float(np.mean([MAUNA_LOA_ANNUAL_CO2[y] for y in co2_years]))
    else:
        co2 = MAUNA_LOA_ANNUAL_CO2[min(MAUNA_LOA_ANNUAL_CO2, key=lambda y: abs(y - args.start_year))]
        print(f"  WARNING: no CO2 table entries in [{args.start_year}, {args.end_year}]; using nearest year's value")
    print(f"Period-mean CO2 (NOAA Mauna Loa, {args.start_year}-{args.end_year}): {co2:.1f} ppm")

    print("Fetching site metadata (elevation, reverse geocode)...")
    site_row = build_site_row(args.location_id, args.lat, args.lon, args.idpl, co2)

    print("Fetching soil texture (SoilGrids) and deriving hydraulic properties (Saxton-Rawls 2006)...")
    soil_row = build_soil_row(args.location_id, args.lat, args.lon)

    print(f"Fetching daily weather {args.start_year}-{args.end_year} (Open-Meteo ERA5 archive)...")
    weather_df = fetch_weather(args.lat, args.lon, args.start_year, args.end_year, args.chunk_years, args.sleep)
    print(f"  {len(weather_df)} days fetched ({weather_df['Date'].iloc[0]} .. {weather_df['Date'].iloc[-1]})")

    site_cols = ["location", "NUTS_ID", "NUTS_NAME", "STATE_NAME", "LATITUDE", "ALTITUDE", "IDPL", "CO2"]
    soil_cols = ["location", "SMDRY", "SMW", "SMFC", "SMO", "CRAIRC", "SMI", "SMLOWI", "RDMSO",
                 "RUNFR", "CFEV", "KSUB", "NMINS", "PMINS", "KMINS", "RTNMINS", "RTPMINS", "RTKMINS"]

    write_csv([site_row], site_cols, out_dir / "site" / "site.csv")
    write_csv([soil_row], soil_cols, out_dir / "soil" / "soil.csv")
    (out_dir / "weather").mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(out_dir / "weather" / f"{args.location_id}.csv", index=False)

    print("\nDone. Wrote:")
    print(f"  {out_dir / 'site' / 'site.csv'}")
    print(f"  {out_dir / 'soil' / 'soil.csv'}")
    print(f"  {out_dir / 'weather' / f'{args.location_id}.csv'}")
    print(
        "\nCaveats: soil.csv's SMDRY/SMW/SMFC/SMO/KSUB are pedotransfer-function estimates "
        "from SoilGrids texture (Saxton-Rawls 2006), not calibrated reference values -- "
        "review before a real run. The remaining soil.csv columns (CRAIRC, RDMSO, RUNFR, "
        "CFEV, N/P/KMINS, RT*MINS) are generic LINTUL5/SIMPLACE defaults reused verbatim "
        "from the Brandenburg reference set, same as that set does across its own 18 "
        "locations. IDPL (sowing day) is a generic default, not location-derived. CO2 is "
        "a period-mean from a static NOAA table, not per-year."
    )


if __name__ == "__main__":
    main()
