#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd, numpy as np

path = "divvy_clean_2024_2026.csv"
df = pd.read_csv(
    path,
    dtype={"rideable_type": "category", "member_casual": "category",
           "start_station_id": "string", "end_station_id": "string"},
    parse_dates=["started_at", "ended_at"],
)

results = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    results.append(bool(ok))

# 1. Size vs. the team's reported figure
check("Row count = 15,851,351", len(df) == 15_851_351, f"(got {len(df):,})")

# 2. Schema
expected = ["ride_id","rideable_type","started_at","ended_at","start_station_name",
            "start_station_id","end_station_name","end_station_id","start_lat",
            "start_lng","end_lat","end_lng","member_casual","ride_length"]
check("Columns match", list(df.columns) == expected)

# 3. Uniqueness and format
check("ride_id unique", df["ride_id"].is_unique, f"({df['ride_id'].duplicated().sum()} repeats)")
check("Full-row duplicates = 0", not df.duplicated().any())
bad_id = ~df["ride_id"].astype(str).str.fullmatch(r"[0-9A-F]{16}")
check("ride_id format", bad_id.sum() == 0, f"({bad_id.sum()} bad)")

# 4. Nulls
for c in ["ride_id","rideable_type","started_at","ended_at","start_lat",
          "start_lng","end_lat","end_lng","member_casual","ride_length"]:
    check(f"No nulls: {c}", df[c].isna().sum() == 0)
for s in ["start", "end"]:
    check(f"{s} station name/id nulls align",
          (df[f"{s}_station_name"].isna() == df[f"{s}_station_id"].isna()).all())

# 5. Categories
print(df["rideable_type"].value_counts(dropna=False))
check("member_casual valid", df["member_casual"].isin(["member","casual"]).all())

# 6. Time logic
check("ended_at > started_at", (df["ended_at"] > df["started_at"]).all())
recalc = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
check("ride_length matches timestamps", np.allclose(df["ride_length"], recalc, atol=0.01))
check("ride_length in [1,180]", df["ride_length"].between(1, 180).all(),
      f"(min {df['ride_length'].min():.2f}, max {df['ride_length'].max():.2f})")
check("No dates after today", df["ended_at"].max() <= pd.Timestamp.now(),
      f"(latest {df['ended_at'].max()})")

monthly = df.groupby(df["started_at"].dt.to_period("M")).size()
print(monthly)
gaps = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M").difference(monthly.index)
check("No missing months", len(gaps) == 0, f"(missing {list(gaps)})")
check("32 months of data", len(monthly) == 32, f"(found {len(monthly)})")

# 7. Coordinates
for la, lo, t in [("start_lat","start_lng","start"), ("end_lat","end_lng","end")]:
    out = ~(df[la].between(41.6, 42.1) & df[lo].between(-88.0, -87.4))
    check(f"{t} coords inside Chicago area", out.sum() == 0, f"({out.sum():,} outside)")

# 8. Stations
for idc, nm in [("start_station_id","start_station_name"), ("end_station_id","end_station_name")]:
    n = df.dropna(subset=[idc]).groupby(idc)[nm].nunique()
    check(f"{idc} maps to one name", (n <= 1).all(), f"({(n>1).sum()} ids multi-named)")

# 9. Plausibility
lat1, lon1, lat2, lon2 = map(np.radians, [df.start_lat, df.start_lng, df.end_lat, df.end_lng])
a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
kmh = 6371 * 2 * np.arcsin(np.sqrt(a)) / (df["ride_length"] / 60)
check("Speed <= 50 km/h", (kmh <= 50).all(), f"({(kmh > 50).sum():,} over)")
same = df["start_station_id"].notna() & (df["start_station_id"] == df["end_station_id"])
print("Same-station rides under 2 min:", (same & (df["ride_length"] < 2)).sum())

print(f"\n{sum(results)}/{len(results)} checks passed")

