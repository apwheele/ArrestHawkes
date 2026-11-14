"""Download Seattle car prowls and arrests; write the analysis file to data/.

    uv run python scripts/01_seattle_data.py
"""

import json
from pathlib import Path

from arresthawkes import seattle

DATA = Path("data")
DATA.mkdir(exist_ok=True)

print("downloading arrests ...", flush=True)
arrests = seattle.download_arrests()
print(f"  {len(arrests):,} arrest rows")

meta = {"start": seattle.START, "end": seattle.END}
for name, fetch in [("carprowl", seattle.download_car_prowls)]:
    print(f"downloading {name} ...", flush=True)
    raw = fetch()
    df = seattle.prepare(raw, arrests)
    df["kind"] = "car prowl"  # a handful also carry a shooting flag
    out = DATA / f"seattle_{name}.csv.zip"
    df.to_csv(out, index=False, compression={"method": "zip", "archive_name": f"seattle_{name}.csv"})
    meta[name] = {
        "offense_rows": len(raw), "reports": df.attrs["n_reports"], "located": df.attrs["located"],
        "n": len(df), "arrested": int(df["arrested"].sum()), "booked": int(df["booked"].sum()),
        "arrest_before_offense": int(df["arrest_before"].sum()),
    }
    print(f"  {len(raw):,} offense rows -> {df.attrs['n_reports']:,} reports -> {len(df):,} located; "
          f"{df['arrested'].sum():,} with an arrest ({df['arrested'].mean():.1%})")
    print("  located by kind (reports, with coordinates):", df.attrs["located"])

json.dump(meta, open(DATA / "seattle_meta.json", "w"), indent=1)
