"""Download and prepare Chicago shootings and arrests.

Two public Socrata tables from the City of Chicago are used:

* Violence Reduction - Victims of Homicides and Non-Fatal Shootings
  (``gumc-mgzr``): one row per victim, with the case (RD) number, date and
  time, whether the victim was shot, whether it was a homicide, and a
  location shifted from the address but kept on the same block.
* Arrests (``dpt3-jri9``): one row
  per arrest with the arrest date and time, the case number for most
  arrests, and up to four charges. The table has no names.

The two join on the case number.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

VICTIMS_URL = "https://data.cityofchicago.org/resource/gumc-mgzr.csv"
ARRESTS_URL = "https://data.cityofchicago.org/resource/dpt3-jri9.csv"

START = "2016-01-01"  # the arrest table's case numbers are sparse before 2016
END = "2026-07-01"  # exclusive; leaves months for arrests

# Charges that make the arrest one of a shooter: homicide, battery,
# firearm discharge, robbery, attempt, assault, hijacking, kidnapping, or a
# warrant (a warrant arrest under the shooting's case number is the arrest
# for that case). Arrests with only gun-possession, drug, trespass or other
# charges may be of the victim or a bystander.
VIOLENT = re.compile(r"MURDER|BATTERY|DISCHARGE|ROBBERY|ATTEMPT|ASSAULT|HOMICIDE|MANSLAUGHTER|HIJACK|KIDNAP|WARRANT",
                     re.I)

# UTM zone 16N, meters
_TO_XY = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)


def _get_csv(url: str, params: dict, timeout: int = 900) -> pd.DataFrame:
    params = dict(params)
    params.setdefault("$limit", 5_000_000)
    res = requests.get(url, params=params, timeout=timeout)
    res.raise_for_status()
    return pd.read_csv(io.StringIO(res.text), dtype=str)


def download_victims(start: str = START) -> pd.DataFrame:
    select = "case_number,date,victimization_primary,incident_primary,gunshot_injury_i,unique_id,latitude,longitude,block"
    return _get_csv(VICTIMS_URL, {"$select": select, "$where": f"gunshot_injury_i = 'YES' AND date >= '{start}'"})


def download_arrests(start: str = START) -> pd.DataFrame:
    select = "cb_no,case_number,arrest_date,charges_statute,charges_description,charges_class"
    return _get_csv(ARRESTS_URL, {"$select": select, "$where": f"case_number IS NOT NULL AND arrest_date >= '{start}'"})


def prepare(victims: pd.DataFrame, arrests: pd.DataFrame, start: str = START, end: str = END,
            violent_only: bool = False) -> pd.DataFrame:
    """One row per shooting incident (case number) with the first arrest.

    The incident time is the first victim's; it is fatal if any victim
    died. An arrest logged more than a day before the shooting cannot belong
    to it and is ignored; one logged up to a day before is treated as made
    at the scene. With ``violent_only`` only arrests with a violent or
    warrant charge count.
    """
    v = victims.copy()
    v["time"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[v["time"].notna() & (v["time"] >= start) & (v["time"] < end)]
    v["fatal"] = v["victimization_primary"] == "HOMICIDE"
    inc = v.sort_values("time").groupby("case_number").agg(
        time=("time", "first"), fatal=("fatal", "max"), n_victims=("unique_id", "size"),
        lat=("latitude", "first"), lon=("longitude", "first"), block_address=("block", "first")).reset_index()
    n_reports = len(inc)
    lat, lon = pd.to_numeric(inc["lat"], errors="coerce"), pd.to_numeric(inc["lon"], errors="coerce")
    ok = lat.between(41.6, 42.1) & lon.between(-88.0, -87.5)
    by_kind = inc.assign(ok=ok).groupby(np.where(inc["fatal"], "fatal", "injury"))["ok"].agg(["size", "sum"])
    inc = inc[ok].copy()
    inc["x"], inc["y"] = [np.round(z, 1) for z in _TO_XY.transform(lon[ok].to_numpy(), lat[ok].to_numpy())]

    a = arrests.copy()
    a["arrest_time"] = pd.to_datetime(a["arrest_date"], errors="coerce")
    a["violent"] = a["charges_description"].fillna("").str.contains(VIOLENT)
    a = a[a["arrest_time"].notna()].merge(inc[["case_number", "time"]], on="case_number")
    a = a[a["arrest_time"] >= a["time"] - pd.Timedelta(days=1)]
    if violent_only:
        a = a[a["violent"]]
    a = a.sort_values("arrest_time")
    first = a.groupby("case_number").agg(arrest_time=("arrest_time", "first"), arrest_type=("charges_description", "first"),
                                         n_arrests=("cb_no", "nunique"), booked=("violent", "max"))
    c = inc.merge(first, left_on="case_number", right_index=True, how="left")
    t0 = pd.Timestamp(start)
    c["t"] = (c["time"] - t0).dt.total_seconds() / 86400
    c["arrest_t"] = (c["arrest_time"] - t0).dt.total_seconds() / 86400
    c["arrest_before"] = c["arrest_t"] < c["t"]
    c["arrest_t"] = np.fmax(c["arrest_t"], c["t"]).where(c["arrest_t"].notna())
    c["arrested"] = c["arrest_t"].notna()
    c["booked"] = c["booked"].fillna(False).astype(bool)  # here: the first arrest had a violent or warrant charge
    c["n_arrests"] = c["n_arrests"].fillna(0).astype(int)
    c["kind"] = np.where(c["fatal"], "fatal", "injury")
    c["report_number"] = c["case_number"]
    c = c.sort_values(["t", "report_number"]).reset_index(drop=True)
    cols = ["report_number", "time", "t", "x", "y", "kind", "n_victims", "block_address", "arrested", "arrest_time",
            "arrest_t", "arrest_type", "booked", "n_arrests", "arrest_before"]
    out = c[cols].copy()
    out.attrs["n_reports"] = n_reports
    out.attrs["located"] = {k: (int(r["size"]), int(r["sum"])) for k, r in by_kind.iterrows()}
    return out
