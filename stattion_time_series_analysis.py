#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df is the cleaned Divvy trip DataFrame
df = df.copy()

df["started_at"] = pd.to_datetime(df["started_at"])
df["start_hour"] = df["started_at"].dt.hour
df["day_of_week"] = (df["started_at"].dt.dayofweek + 1) % 7 + 1
df["day_type"] = df["started_at"].dt.dayofweek.map(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

# 1. Top casual-rider starting stations
casual = df[df["member_casual"] == "casual"].copy()

top_stations = (
    casual.dropna(subset=["start_station_name"])
    .groupby("start_station_name")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 casual starting stations:")
print(top_stations)

top_stations.sort_values().plot(
    kind="barh", figsize=(10, 6)
)
plt.title("Top 10 Starting Stations for Casual Riders")
plt.xlabel("Number of Trip Starts")
plt.ylabel("Station")
plt.tight_layout()
plt.show()

# 2. Hourly activity by rider category
hourly = (
    df.groupby(["start_hour", "member_casual"])
    .size()
    .reset_index(name="trip_count")
)

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
plt.tight_layout()
plt.show()

# 3. Station-hour heatmap for top casual stations
heatmap_data = (
    casual[casual["start_station_name"].isin(top_stations.index)]
    .groupby(["start_station_name", "start_hour"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(24), fill_value=0)
)

plt.figure(figsize=(13, 7))
sns.heatmap(heatmap_data, cmap="Blues")
plt.title("Casual-Rider Trip Starts by Station and Hour")
plt.xlabel("Starting Hour")
plt.ylabel("Starting Station")
plt.tight_layout()
plt.show()

