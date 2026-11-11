#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\NEW-PC\Downloads\divvy_clean_2024_2026.csv")

df = df.copy()

df["started_at"] = pd.to_datetime(
    df["started_at"], errors="coerce"
)

# Remove rows without valid starting timestamps
df = df.dropna(subset=["started_at"])

# Extract starting hour (0 to 23)
df["start_hour"] = df["started_at"].dt.hour

# Sunday = 1, Monday = 2, ..., Saturday = 7
df["day_of_week"] = (
    (df["started_at"].dt.dayofweek + 1) % 7 + 1
)

# Classify weekdays and weekends
df["day_type"] = df["started_at"].dt.dayofweek.map(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

# Standardize membership type
df["member_casual"] = (
    df["member_casual"]
    .astype("string")
    .str.strip()
    .str.lower()
)

casual = df[
    df["member_casual"] == "casual"
].copy()

top_stations = (
    casual.dropna(subset=["start_station_name"])
    .groupby("start_station_name")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 CASUAL STARTING STATIONS")
print(top_stations)

plt.figure(figsize=(11, 6))

top_stations.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Starting Stations for Casual Riders")
plt.xlabel("Number of Trip Starts")
plt.ylabel("Starting Station")
plt.tight_layout()
plt.show()

hourly = (
    df.groupby(["start_hour", "member_casual"])
    .size()
    .reset_index(name="trip_count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly,
    x="start_hour",
    y="trip_count",
    hue="member_casual",
    marker="o"
)

plt.title("Hourly Ridership by Membership Type")
plt.xlabel("Starting Hour")
plt.ylabel("Number of Trip Starts")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Count trips by day type and membership
day_summary = (
    df.groupby(["day_type", "member_casual"])
    .size()
    .reset_index(name="trip_count")
)

print("\nWEEKDAY VS WEEKEND RIDERSHIP")
print(day_summary)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=day_summary,
    x="day_type",
    y="trip_count",
    hue="member_casual"
)

plt.title("Weekday vs Weekend Ridership")
plt.xlabel("Day Type")
plt.ylabel("Total Trip Starts")
plt.tight_layout()
plt.show()

heatmap_data = (
    casual[
        casual["start_station_name"].isin(
            top_stations.index
        )
    ]
    .groupby(["start_station_name", "start_hour"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(24), fill_value=0)
)

plt.figure(figsize=(14, 8))

sns.heatmap(
    heatmap_data,
    cmap="Blues",
    annot=False
)

plt.title(
    "Casual-Rider Trip Starts by Station and Hour"
)
plt.xlabel("Starting Hour")
plt.ylabel("Starting Station")
plt.tight_layout()
plt.show()


peak_hours = (
    casual.groupby("start_hour")
    .size()
    .sort_values(ascending=False)
)

print("\nTOP 5 PEAK HOURS FOR CASUAL RIDERS")
print(peak_hours.head(5))

station_counts = (
    df.dropna(subset=["start_station_name"])
    .groupby(["start_station_name", "member_casual"])
    .size()
    .unstack(fill_value=0)
)

for category in ["casual", "member"]:
    if category not in station_counts.columns:
        station_counts[category] = 0

station_counts["total_trips"] = (
    station_counts["casual"]
    + station_counts["member"]
)

station_counts["casual_percentage"] = (
    station_counts["casual"]
    / station_counts["total_trips"].replace(0, float("nan"))
    * 100
)

# Rank by casual trip volume
station_ranking = station_counts.sort_values(
    "casual", ascending=False
).head(10)

print("\nTOP 10 STATIONS AND CASUAL RIDER SHARE")
print(
    station_ranking[
        ["casual", "member",
         "total_trips", "casual_percentage"]
    ].round(2)
)

print("\nStation and Time Analysis completed!")


# In[ ]:




