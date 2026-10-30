"""Build a per-season torchcrop dataset from NASA POWER daily weather.

A companion to ``fetch_torchcrop_dataset.py`` (Open-Meteo/ERA5) for sites
whose other inputs already come from NASA POWER, so that the crop model and
the disease model are driven by the same weather. It writes the per-season
layout the Indiana notebook reads, one "location" per sowing year::

    <out-dir>/
        site/site.csv          one row per season: location (= sowing year),
                                NUTS_ID, NUTS_NAME, STATE_NAME, LATITUDE,
                                ALTITUDE, IDPL, CO2
        soil/soil.csv          one row per season (same soil every season)
        weather/<year>.csv     daily, from 1 January of the sowing year for
                                ``--season-days`` days: Date, TempMean, TempMin,
                                TempMax, Radiation, Precipitation, VapPressure,
                                Windspeed

Weather (NASA POWER daily API, community AG, https://power.larc.nasa.gov):

    TempMin/TempMax   T2M_MIN/T2M_MAX [°C], MERRA-2
    TempMean          (TempMin + TempMax) / 2, as torchcrop's reader uses
    Radiation         ALLSKY_SFC_SW_DWN [MJ m-2 d-1], CERES SYN1deg, written
                      in kJ m-2 d-1 like the Open-Meteo tool
    Precipitation     PRECTOTCORR [mm d-1], MERRA-2
    VapPressure       actual vapour pressure from the dew point T2MDEW,
                      0.6108 exp(17.27 Td / (Td + 237.3)) [kPa] (FAO-56 eq. 14)
    Windspeed         WS2M [m s-1], MERRA-2, already at 2 m

Site and soil reuse ``fetch_torchcrop_dataset.py``: elevation (Open-Meteo),
soil texture (SoilGrids) through the Saxton-Rawls (2006) pedotransfer
function plus the generic LINTUL5 soil defaults, and the NOAA Mauna Loa
annual CO2 of each sowing year.

Usage:
    python tools/fetch_power_torchcrop_dataset.py \\
        --lat 30.6150167 --lon 31.5145167 --sow-years 2013 2014 --idpl 300 \\
        --out-dir notebooks/Sharkiya/SharkiyaData
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_torchcrop_dataset as base  # noqa: E402

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS2M", "T2MDEW"]


def fetch_power_daily(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily NASA POWER records between two dates, one row per day."""
    query = urllib.parse.urlencode({
        "parameters": ",".join(POWER_PARAMETERS), "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "CSV",
    })
    req = urllib.request.Request(f"{POWER_URL}?{query}", headers={"User-Agent": base.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8").replace("\r\n", "\n")
    df = pd.read_csv(io.StringIO(text.split("-END HEADER-\n", 1)[1]))
    if (df[POWER_PARAMETERS] == -999).any().any():
        raise ValueError("NASA POWER returned missing values (-999)")
    df["Date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + pd.to_timedelta(df["DOY"] - 1, unit="D")
    return df


def to_torchcrop(df: pd.DataFrame) -> pd.DataFrame:
    """NASA POWER daily records in the torchcrop weather layout."""
    td = df["T2MDEW"].astype(float)
    return pd.DataFrame({
        "Date": df["Date"].dt.strftime("%d/%m/%Y"),
        "TempMean": ((df["T2M_MIN"] + df["T2M_MAX"]) / 2).round(2),
        "TempMin": df["T2M_MIN"],
        "TempMax": df["T2M_MAX"],
        "Radiation": (df["ALLSKY_SFC_SW_DWN"] * 1000.0).round(0),
        "Precipitation": df["PRECTOTCORR"],
        "VapPressure": (0.6108 * np.exp(17.27 * td / (td + 237.3))).round(3),
        "Windspeed": df["WS2M"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--sow-years", type=int, nargs="+", required=True)
    parser.add_argument("--idpl", type=int, required=True, help="Sowing day of year, every season")
    parser.add_argument("--season-days", type=int, default=570,
                        help="Days per season file, from 1 January of the sowing year")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    first = pd.Timestamp(f"{min(args.sow_years)}-01-01")
    last = pd.Timestamp(f"{max(args.sow_years)}-01-01") + pd.Timedelta(days=args.season_days - 1)
    print(f"Fetching NASA POWER daily weather {first.date()} .. {last.date()}")
    weather = to_torchcrop(fetch_power_daily(args.lat, args.lon, first, last))
    weather_dates = pd.to_datetime(weather["Date"], format="%d/%m/%Y")

    print("Fetching elevation, place names and soil texture...")
    site_template = base.build_site_row(0, args.lat, args.lon, args.idpl, 0.0)
    soil_template = base.build_soil_row(0, args.lat, args.lon)

    site_rows, soil_rows = [], []
    for year in args.sow_years:
        start = pd.Timestamp(f"{year}-01-01")
        season = weather[(weather_dates >= start) & (weather_dates < start + pd.Timedelta(days=args.season_days))]
        out = args.out_dir / "weather" / f"{year}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        season.to_csv(out, index=False)
        print(f"  {out}  ({len(season)} days)")
        site_rows.append({**site_template, "location": year, "CO2": base.MAUNA_LOA_ANNUAL_CO2[year]})
        soil_rows.append({**soil_template, "location": year})

    site_cols = ["location", "NUTS_ID", "NUTS_NAME", "STATE_NAME", "LATITUDE", "ALTITUDE", "IDPL", "CO2"]
    soil_cols = ["location", "SMDRY", "SMW", "SMFC", "SMO", "CRAIRC", "SMI", "SMLOWI", "RDMSO",
                 "RUNFR", "CFEV", "KSUB", "NMINS", "PMINS", "KMINS", "RTNMINS", "RTPMINS", "RTKMINS"]
    base.write_csv(site_rows, site_cols, args.out_dir / "site" / "site.csv")
    base.write_csv(soil_rows, soil_cols, args.out_dir / "soil" / "soil.csv")
    print(f"  {args.out_dir / 'site' / 'site.csv'}\n  {args.out_dir / 'soil' / 'soil.csv'}")


if __name__ == "__main__":
    main()
