"""Download and prepare Seattle Police Department crime and arrest data.

Two public Socrata tables are used:

* SPD Crime Data: 2008-Present (``tazs-3rd5``): one row per offense, with the
  report number, offense date and time, a shooting flag, and a location
  masked to the hundred block.
* SPD Arrest Data (``9bjs-7a7w``): one row per arrest charge from 2019 on,
  with the arrest date and time and the report number of the case.

The two join on the report number, which is written ``2024-123456`` in the
crime table and ``20240000123456`` in the arrest table.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

CRIME_URL = "https://data.seattle.gov/resource/tazs-3rd5.csv"
ARREST_URL = "https://cos-data.seattle.gov/resource/9bjs-7a7w.csv"

START = "2019-01-01"  # first full year of the arrest table
END = "2026-07-01"  # exclusive; leaves months for late-entered arrests

SHOOTING_TYPES = {
    "Shooting (Fatal Injury)": "fatal",
    "Shooting (Non-Fatal Injury)": "injury",
    "Shots Fired (Eyewitness/Casings/Property Damage)": "shots fired",
}
SEVERITY = {"fatal": 0, "injury": 1, "shots fired": 2}
CAR_PROWL = "Theft From Motor Vehicle"

# UTM zone 10N, meters
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)


def _get_csv(url: str, params: dict, timeout: int = 900) -> pd.DataFrame:
    params = dict(params)
    params.setdefault("$limit", 1_000_000)
    res = requests.get(url, params=params, timeout=timeout)
    res.raise_for_status()
    return pd.read_csv(io.StringIO(res.text), dtype=str)


def download_shootings(start: str = START, end: str = END) -> pd.DataFrame:
    where = f"shooting_type_group != '-' AND offense_date >= '{start}' AND offense_date < '{end}'"
    return _get_csv(CRIME_URL, {"$where": where})


def download_car_prowls(start: str = START, end: str = END) -> pd.DataFrame:
    where = (
        f"nibrs_offense_code_description = '{CAR_PROWL}'"
        f" AND offense_date >= '{start}' AND offense_date < '{end}'"
    )
    return _get_csv(CRIME_URL, {"$where": where})


def download_arrests() -> pd.DataFrame:
    select = "arrest_number,arrest_occurred_date_time,arrest_reported_date_time,arrest_type,nibrs_description,report_number"
    return _get_csv(ARREST_URL, {"$select": select})


def first_arrests(arrests: pd.DataFrame) -> pd.DataFrame:
    """One row per report number: the earliest arrest tied to the case."""
    a = arrests.copy()
    rn = a["report_number"].fillna("")
    a["report_number"] = rn.str[:4] + "-" + rn.str[-6:]
    a["arrest_time"] = pd.to_datetime(a["arrest_occurred_date_time"], errors="coerce")
    a = a[a["arrest_time"].notna() & (rn.str.len() == 14)]
    # "Booked" types put the person in jail; the rest identify or charge a
    # suspect without custody ("CBO-not booked", "Identify and Release-Not
    # Booked", citations, declined at the jail, other agency's arrest).
    typ = a["arrest_type"].fillna("")
    a["booked"] = typ.str.contains("Booked") & ~typ.str.contains("not booked", case=False)
    a = a.sort_values(["report_number", "arrest_time", "booked"], ascending=[True, True, False])
    n = a.groupby("report_number")["arrest_number"].nunique().rename("n_arrests")
    first = a.drop_duplicates("report_number").set_index("report_number")
    return first[["arrest_time", "arrest_type", "booked"]].join(n).reset_index()


def prepare(crime: pd.DataFrame, arrests: pd.DataFrame, start: str = START, end: str = END) -> pd.DataFrame:
    """One row per report with projected coordinates and the first arrest.

    Rows with a redacted location are dropped (``attrs['n_redacted']``
    records how many). Times are days since ``start``; an arrest logged
    before the offense time is treated as made at the scene.
    """
    c = crime.copy()
    c["time"] = pd.to_datetime(c["offense_date"], errors="coerce")
    c = c[c["time"].notna() & (c["time"] >= start) & (c["time"] < end)]
    c["kind"] = c["shooting_type_group"].map(SHOOTING_TYPES).fillna("car prowl")
    c["sev"] = c["kind"].map(SEVERITY).fillna(9)
    c = c.sort_values(["report_number", "sev", "time"]).drop_duplicates("report_number")
    n_reports = len(c)

    lon = pd.to_numeric(c["longitude"], errors="coerce")
    lat = pd.to_numeric(c["latitude"], errors="coerce")
    ok = lon.between(-123, -121.5) & lat.between(47.2, 48.0)
    by_kind = c.assign(ok=ok).groupby("kind")["ok"].agg(["size", "sum"])
    c = c[ok].copy()
    x, y = _TO_UTM.transform(lon[ok].to_numpy(), lat[ok].to_numpy())
    c["x"], c["y"] = np.round(x, 1), np.round(y, 1)

    c = c.merge(first_arrests(arrests), on="report_number", how="left")
    t0 = pd.Timestamp(start)
    c["t"] = (c["time"] - t0).dt.total_seconds() / 86400
    c["arrest_t"] = (c["arrest_time"] - t0).dt.total_seconds() / 86400
    c["arrest_before"] = c["arrest_t"] < c["t"]
    c["arrest_t"] = np.fmax(c["arrest_t"], c["t"]).where(c["arrest_t"].notna())
    c["arrested"] = c["arrest_t"].notna()
    c["booked"] = c["booked"].fillna(False).astype(bool)
    c["n_arrests"] = c["n_arrests"].fillna(0).astype(int)
    c = c.sort_values(["t", "report_number"]).reset_index(drop=True)
    cols = ["report_number", "time", "t", "x", "y", "kind", "block_address", "beat", "arrested",
            "arrest_time", "arrest_t", "arrest_type", "booked", "n_arrests", "arrest_before"]
    out = c[cols].copy()
    out.attrs["n_reports"] = n_reports
    out.attrs["located"] = {k: (int(r["size"]), int(r["sum"])) for k, r in by_kind.iterrows()}
    return out


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time", "arrest_time"], dtype={"report_number": str})


def jitter(df: pd.DataFrame, sd: float, seed: int = 0):
    """Coordinates with Gaussian noise of ``sd`` meters added to each axis.

    SPD publishes the hundred block, so many reports share a point; exact
    ties make a continuous spatial kernel degenerate.
    """
    rng = np.random.default_rng(seed)
    return (df["x"].to_numpy() + rng.normal(0, sd, len(df)),
            df["y"].to_numpy() + rng.normal(0, sd, len(df)))
