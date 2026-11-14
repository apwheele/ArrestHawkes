"""Download Chicago shootings and arrests; write the analysis files to data/.

    uv run python scripts/02_chicago_data.py

Writes data/chicago_shootings.csv.zip (first arrest under the shooting's
case number, the main specification), data/chicago_shootings_violent.csv.zip
(first arrest with a violent or warrant charge, a sensitivity check), and
data/chicago_meta.json, which includes the arrest share by year that the
paper compares with published Chicago arrest rates.
"""

import json
from pathlib import Path

from arresthawkes import chicago

DATA = Path("data")
DATA.mkdir(exist_ok=True)
print("downloading shooting victims ...", flush=True)
victims = chicago.download_victims()
print(f"  {len(victims):,} victim rows")
print("downloading arrests ...", flush=True)
arrests = chicago.download_arrests()
print(f"  {len(arrests):,} arrests with a case number")

meta = {"start": chicago.START, "end": chicago.END, "victim_rows": len(victims), "arrest_rows": len(arrests)}
for name, violent in [("chicago_shootings", False), ("chicago_shootings_violent", True)]:
    df = chicago.prepare(victims, arrests, violent_only=violent)
    df.to_csv(DATA / f"{name}.csv.zip", index=False, compression={"method": "zip", "archive_name": f"{name}.csv"})
    by_year = df.assign(year=df["time"].dt.year).groupby(["year", "kind"])["arrested"].mean().unstack()
    meta[name] = {
        "reports": df.attrs["n_reports"], "located": df.attrs["located"], "n": len(df),
        "fatal": int((df["kind"] == "fatal").sum()), "arrested": int(df["arrested"].sum()),
        "arrested_fatal": int((df["arrested"] & (df["kind"] == "fatal")).sum()),
        "arrested_injury": int((df["arrested"] & (df["kind"] == "injury")).sum()),
        "first_arrest_violent": int((df["arrested"] & df["booked"]).sum()),
        "arrest_before_offense": int(df["arrest_before"].sum()),
        "arrest_share_by_year": {str(y): {k: round(float(v), 3) for k, v in r.items()} for y, r in by_year.iterrows()},
    }
    print(f"{name}: {len(df):,} incidents, {int(df['arrested'].sum()):,} with an arrest ({df['arrested'].mean():.1%})")
    print(by_year.round(3).to_string())
json.dump(meta, open(DATA / "chicago_meta.json", "w"), indent=1)
