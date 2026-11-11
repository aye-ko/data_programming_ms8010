#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option("display.float_format", "{:,.2f}".format)
os.makedirs("figures", exist_ok=True)

# one color per rider group, used the same way in every chart
COLORS = {"member": "#2a78d6", "casual": "#eb6834"}
GRAY = "#8a8985"

plt.rcParams.update({
    "figure.figsize": (11, 4.5), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "lines.linewidth": 2, "legend.frameon": False,
})
thousands = mtick.FuncFormatter(lambda x, _: f"{x/1000:,.0f}k")
comma = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")

COLS = ["ride_id", "rideable_type", "started_at", "start_station_name",
        "member_casual", "ride_length"]

# looks for the cleaned file in the places the earlier notebooks saved it
candidates = ["data/divvy_clean_2024_2026.parquet", "divvy_clean_2024_2026.parquet",
              "data/divvy_clean_2024_2026.csv", "divvy_clean_2024_2026.csv"]
path = next(p for p in candidates if os.path.exists(p))
print("Loading", path)

if path.endswith(".parquet"):
    df = pd.read_parquet(path, columns=COLS)
else:
    df = pd.read_csv(path, usecols=COLS, parse_dates=["started_at"],
                     dtype={"rideable_type": "category", "member_casual": "category"})

df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
df["member_casual"] = df["member_casual"].astype("category")

before = len(df)
df = df.drop_duplicates(subset="ride_id", keep="first")
print(f"Rows loaded: {before:,} | repeated ride_ids dropped: {before - len(df):,} | rows used: {len(df):,}")
print("Date range:", df["started_at"].min(), "to", df["started_at"].max())

ts = df["started_at"]
df["date"] = ts.dt.normalize()
df["year"] = ts.dt.year.astype("int16")
df["month"] = ts.dt.month.astype("int8")
df["year_month"] = ts.dt.to_period("M")
df["hour"] = ts.dt.hour.astype("int8")
df["day_of_week"] = ((ts.dt.dayofweek + 1) % 7 + 1).astype("int8")   # 1=Sun ... 7=Sat
df["is_weekend"] = df["day_of_week"].isin([1, 7])

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
df["season"] = pd.Categorical(df["month"].map(SEASON), categories=SEASON_ORDER, ordered=True)

df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

DAY_NAMES = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

df[["started_at", "member_casual", "day_of_week", "season", "month", "month_sin", "month_cos"]].head()

monthly = (df.groupby(["year_month", "member_casual"], observed=True).size()
             .unstack("member_casual").sort_index())
monthly["total"] = monthly.sum(axis=1)
monthly["casual_share"] = monthly["casual"] / monthly["total"]

fig, ax = plt.subplots()
x = monthly.index.to_timestamp()
for g in ["member", "casual"]:
    ax.plot(x, monthly[g], color=COLORS[g], marker="o", markersize=4, label=g.title())
    ax.annotate(g.title(), (x[-1], monthly[g].iloc[-1]), xytext=(6, 0),
                textcoords="offset points", color="#0b0b0b", va="center", fontsize=10)
ax.yaxis.set_major_formatter(thousands)
ax.set_title("Rides per month, casual vs member (2024-2026)")
ax.set_ylabel("Rides")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("figures/01_monthly_rides.png", bbox_inches="tight"); plt.show()

monthly.assign(casual_share=monthly["casual_share"].map("{:.1%}".format))

# casual share of all rides, averaged by calendar month across years
by_month = df.groupby(["month", "member_casual"], observed=True).size().unstack()
n_years = df.groupby("month")["year"].nunique()
avg_month = by_month.div(n_years, axis=0)              # average rides per calendar month
avg_month["casual_share"] = avg_month["casual"] / avg_month.sum(axis=1)

fig, ax = plt.subplots(figsize=(11, 3.8))
bars = ax.bar(MONTH_NAMES, avg_month["casual_share"] * 100, color=COLORS["casual"], width=0.6)
ax.axhline(df["member_casual"].eq("casual").mean() * 100, color=GRAY, linestyle="--", linewidth=1)
ax.text(11.4, df["member_casual"].eq("casual").mean() * 100, " overall", color="#52514e", va="bottom")
for b, v in zip(bars, avg_month["casual_share"]):
    if v == avg_month["casual_share"].max() or v == avg_month["casual_share"].min():
        ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center")
ax.set_ylabel("Casual share of rides (%)")
ax.set_title("Casual riders make up a bigger share of rides in summer")
plt.tight_layout(); plt.savefig("figures/02_casual_share_by_month.png", bbox_inches="tight"); plt.show()

season_tbl = (df.groupby(["season", "member_casual"], observed=True)
                .agg(rides=("ride_id", "size"), avg_ride_min=("ride_length", "mean"),
                     median_ride_min=("ride_length", "median"))
                .unstack("member_casual"))
season_tbl[("share_of_year", "casual")] = (season_tbl[("rides", "casual")]
                                           / season_tbl[("rides", "casual")].sum())
season_tbl[("share_of_year", "member")] = (season_tbl[("rides", "member")]
                                           / season_tbl[("rides", "member")].sum())
season_tbl

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
w = 0.38
xs = np.arange(len(SEASON_ORDER))
for i, g in enumerate(["member", "casual"]):
    axes[0].bar(xs + (i - 0.5) * w, season_tbl[("share_of_year", g)] * 100, w,
                color=COLORS[g], label=g.title(), edgecolor="white", linewidth=2)
    axes[1].bar(xs + (i - 0.5) * w, season_tbl[("avg_ride_min", g)], w,
                color=COLORS[g], label=g.title(), edgecolor="white", linewidth=2)
for ax in axes:
    ax.set_xticks(xs, SEASON_ORDER)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)   # headroom so the legend doesn't cover bars
    ax.legend(loc="upper left", ncol=2)
axes[0].set_title("Share of each group's rides by season")
axes[0].set_ylabel("% of the group's rides")
axes[1].set_title("Average ride length by season")
axes[1].set_ylabel("Minutes")
plt.tight_layout(); plt.savefig("figures/03_seasons.png", bbox_inches="tight"); plt.show()

# average rides per day for each day of week (divide by number of such days, so busy months don't skew it)
days_count = df.groupby("day_of_week")["date"].nunique()
dow = df.groupby(["day_of_week", "member_casual"], observed=True).size().unstack().div(days_count, axis=0)

hourly = df.groupby(["is_weekend", "hour", "member_casual"], observed=True).size().unstack()
n_wkend = df.loc[df["is_weekend"], "date"].nunique()
n_wkday = df.loc[~df["is_weekend"], "date"].nunique()
hourly.loc[True] = (hourly.loc[True] / n_wkend).values
hourly.loc[False] = (hourly.loc[False] / n_wkday).values

fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
xs = np.arange(7); w = 0.38
for i, g in enumerate(["member", "casual"]):
    axes[0].bar(xs + (i - 0.5) * w, dow[g], w, color=COLORS[g], label=g.title(),
                edgecolor="white", linewidth=2)
axes[0].set_xticks(xs, [DAY_NAMES[d] for d in dow.index])
axes[0].yaxis.set_major_formatter(comma)
axes[0].set_title("Avg rides per day")
axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.25)
axes[0].legend(loc="upper left", ncol=2, fontsize=9)

for ax, wk, title in [(axes[1], False, "Weekday: avg rides per hour"),
                      (axes[2], True, "Weekend: avg rides per hour")]:
    for g in ["member", "casual"]:
        ax.plot(hourly.loc[wk].index, hourly.loc[wk][g], color=COLORS[g], label=g.title())
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_title(title)
    ax.legend()
axes[2].sharey(axes[1])
plt.tight_layout(); plt.savefig("figures/04_day_and_hour.png", bbox_inches="tight"); plt.show()

dow.rename(index=DAY_NAMES).assign(casual_share=lambda t: t["casual"] / t.sum(axis=1))

# When exactly are casual riders out? Average casual rides per hour, by day of week and hour (peak season Jun-Aug)
peak = df[df["month"].isin([6, 7, 8]) & (df["member_casual"] == "casual")]
heat = peak.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
heat = heat.div(peak.groupby("day_of_week")["date"].nunique(), axis=0)

fig, ax = plt.subplots(figsize=(12, 3.8))
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
im = ax.imshow(heat.values, aspect="auto", cmap=blues)
ax.set_yticks(range(7), [DAY_NAMES[d] for d in heat.index])
ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
ax.grid(False)
ax.set_title("Casual rides per hour in summer (Jun-Aug): darker = busier")
fig.colorbar(im, ax=ax, label="Avg casual rides / hour", pad=0.01)
plt.tight_layout(); plt.savefig("figures/05_casual_heatmap_summer.png", bbox_inches="tight"); plt.show()

top_slots = heat.stack().sort_values(ascending=False).head(10).rename("avg_casual_rides_per_hour").reset_index()
top_slots["day_of_week"] = top_slots["day_of_week"].map(DAY_NAMES)
top_slots

st = df.dropna(subset=["start_station_name"])
station = st.groupby(["start_station_name", "member_casual"], observed=True).size().unstack(fill_value=0)
station["total"] = station.sum(axis=1)
station["casual_share"] = station["casual"] / station["total"]

top_casual = station.sort_values("casual", ascending=False).head(15)

# same ranking for the summer peak only
st_peak = st[st["month"].isin([6, 7, 8]) & (st["member_casual"] == "casual")]
top_peak = st_peak["start_station_name"].value_counts().head(15).rename("casual_rides_jun_aug")

print(f"Top 15 stations = {top_casual['casual'].sum() / station['casual'].sum():.1%} of all casual rides with a station")
top_casual.join(top_peak)

plot_df = top_casual.sort_values("casual")
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(plot_df.index, plot_df["casual"], color=COLORS["casual"], height=0.65)
for y, (c, s) in enumerate(zip(plot_df["casual"], plot_df["casual_share"])):
    ax.text(c, y, f"  {s:.0%} casual", va="center", color="#52514e", fontsize=9)
ax.xaxis.set_major_formatter(thousands)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Casual rides started (Jan 2024 - Sep 2026)")
ax.set_title("Top 15 start stations for casual riders")
plt.tight_layout(); plt.savefig("figures/06_top_casual_stations.png", bbox_inches="tight"); plt.show()

daily = (df[df["member_casual"] == "casual"].groupby("date").size()
           .rename("casual_rides"))
all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
daily = daily.reindex(all_days, fill_value=0).to_frame()
daily.index.name = "date"
daily["member_rides"] = (df[df["member_casual"] == "member"].groupby("date").size()
                           .reindex(all_days, fill_value=0))

holidays = USFederalHolidayCalendar().holidays(start="2024-01-01", end="2027-12-31")
START = pd.Timestamp("2024-01-01")

def make_features(dates):
    dates = pd.DatetimeIndex(dates)
    X = pd.DataFrame(index=dates)
    m = dates.month
    X["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    doy = dates.dayofyear / 365.25
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    X["doy_sin2"], X["doy_cos2"] = np.sin(4 * np.pi * doy), np.cos(4 * np.pi * doy)
    dow = (dates.dayofweek + 1) % 7 + 1                       # 1=Sun ... 7=Sat
    for d in range(1, 8):
        X[f"dow_{d}"] = (dow == d).astype(int)
    X["is_weekend"] = np.isin(dow, [1, 7]).astype(int)
    X["is_holiday"] = dates.isin(holidays).astype(int)
    X["trend"] = (dates - START).days / 365.25
    return X

X_all = make_features(daily.index)
y_all = daily["casual_rides"]
print(daily.shape, "days |", X_all.shape[1], "features")
X_all.head()

train = daily.index < "2025-01-01"
valid = (daily.index >= "2025-01-01") & (daily.index < "2026-01-01")
train_full = daily.index < "2026-01-01"
test = daily.index >= "2026-01-01"
print(f"train {train.sum()} days | valid {valid.sum()} | final train {train_full.sum()} | test {test.sum()}")

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true > 0
    return {"RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])),
            "R2": r2_score(y_true, y_pred)}

class SeasonalBaseline:
    # average of the training days with the same month and day of week
    def fit(self, X, y):
        key = pd.Series(list(zip(X.index.month, X.index.dayofweek)), index=X.index)
        self.table = y.groupby(key.values).mean()
        self.fallback = y.mean()
        return self
    def predict(self, X):
        keys = zip(X.index.month, X.index.dayofweek)
        return np.array([self.table.get(k, self.fallback) for k in keys])

class LogLinear:
    # linear regression on log(1 + rides); predictions converted back to rides
    def __init__(self, cols): self.cols, self.m = cols, LinearRegression()
    def fit(self, X, y): self.m.fit(X[self.cols], np.log1p(y)); return self
    def predict(self, X): return np.expm1(self.m.predict(X[self.cols])).clip(0)

LIN_COLS = [c for c in X_all.columns if c not in ("dow_1", "is_weekend")]   # drop one dummy to avoid collinearity
TREE_COLS = [c for c in X_all.columns if c != "trend"]   # trees can't extrapolate a trend, so leave it out

def make_models():
    return {
        "Baseline (month x weekday avg)": SeasonalBaseline(),
        "Linear regression (wave)": LogLinear(LIN_COLS),
        "Gradient boosting": GBWrapper(),
    }

class GBWrapper:
    def __init__(self):
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=10, l2_regularization=1.0, random_state=42)
    def fit(self, X, y): self.m.fit(X[TREE_COLS], y); return self
    def predict(self, X): return self.m.predict(X[TREE_COLS]).clip(0)

val_rows = []
for name, model in make_models().items():
    model.fit(X_all[train], y_all[train])
    val_rows.append({"model": name, **metrics(y_all[valid], model.predict(X_all[valid]))})
val_results = pd.DataFrame(val_rows).set_index("model")
val_results.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "MAPE": "{:.1%}", "R2": "{:.3f}"})

fitted, preds, test_rows = {}, {}, []
for name, model in make_models().items():
    model.fit(X_all[train_full], y_all[train_full])
    fitted[name] = model
    preds[name] = model.predict(X_all[test])
    test_rows.append({"model": name, **metrics(y_all[test], preds[name])})
test_results = pd.DataFrame(test_rows).set_index("model")

base_rmse = test_results.loc["Baseline (month x weekday avg)", "RMSE"]
test_results["RMSE vs baseline"] = test_results["RMSE"] / base_rmse - 1
best_name = test_results.drop("Baseline (month x weekday avg)")["RMSE"].idxmin()
print("Average casual rides per day in the 2026 test period:", f"{y_all[test].mean():,.0f}")
print("Best model on 2026:", best_name)
test_results.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "MAPE": "{:.1%}",
                           "R2": "{:.3f}", "RMSE vs baseline": "{:+.1%}"})

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(daily.index[test], y_all[test], color=GRAY, linewidth=1, label="Actual")
ax.plot(daily.index[test], pd.Series(preds[best_name], index=daily.index[test]).rolling(7, center=True).mean(),
        color=COLORS["casual"], label=f"{best_name} (7-day avg)")
ax.plot(daily.index[test], pd.Series(y_all[test]).rolling(7, center=True).mean(),
        color="#0b0b0b", linewidth=1.2, linestyle="--", label="Actual (7-day avg)")
ax.yaxis.set_major_formatter(thousands)
ax.set_ylabel("Casual rides per day")
ax.set_title("2026 test period: predicted vs actual casual rides")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("figures/07_test_actual_vs_pred.png", bbox_inches="tight"); plt.show()

# Monthly totals are what an ad plan is budgeted on, so check error at that level too
m_actual = y_all[test].groupby(daily.index[test].to_period("M")).sum()
m_pred = pd.Series(preds[best_name], index=daily.index[test]).groupby(daily.index[test].to_period("M")).sum()
monthly_eval = pd.DataFrame({"actual": m_actual, "predicted": m_pred.round()})
monthly_eval["error_%"] = (monthly_eval["predicted"] / monthly_eval["actual"] - 1)
print(f"Monthly-total MAPE: {monthly_eval['error_%'].abs().mean():.1%}")
monthly_eval.style.format({"actual": "{:,.0f}", "predicted": "{:,.0f}", "error_%": "{:+.1%}"})

# Residuals: where does the model miss?
res = pd.DataFrame({"actual": y_all[test].values, "pred": preds[best_name]}, index=daily.index[test])
res["residual"] = res["actual"] - res["pred"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(res["pred"], res["actual"], s=14, color=COLORS["casual"], alpha=0.6, edgecolor="white", linewidth=0.5)
lim = [0, max(res["actual"].max(), res["pred"].max()) * 1.05]
axes[0].plot(lim, lim, color=GRAY, linestyle="--", linewidth=1)
axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].xaxis.set_major_formatter(thousands); axes[0].yaxis.set_major_formatter(thousands)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Predicted vs actual (each dot = one day)")

axes[1].hist(res["residual"], bins=40, color=COLORS["casual"], edgecolor="white", linewidth=1)
axes[1].axvline(0, color=GRAY, linestyle="--", linewidth=1)
axes[1].xaxis.set_major_formatter(thousands)
axes[1].set_xlabel("Actual - predicted (rides)")
axes[1].set_title("Residuals")
plt.tight_layout(); plt.savefig("figures/08_residuals.png", bbox_inches="tight"); plt.show()

print("Largest misses (likely weather or events the calendar can't see):")
res.reindex(res["residual"].abs().sort_values(ascending=False).index).head(8)

# Which features matter most? (permutation importance on the test set: how much RMSE gets worse when a feature is shuffled)
best = fitted[best_name]
cols = best.cols if hasattr(best, "cols") else TREE_COLS
X_test, y_test = X_all[test].copy(), y_all[test]
base = np.sqrt(mean_squared_error(y_test, best.predict(X_test)))
rng = np.random.default_rng(42)
imp = {}
for c in cols:
    scores = []
    for _ in range(10):
        Xp = X_test.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        scores.append(np.sqrt(mean_squared_error(y_test, best.predict(Xp))) - base)
    imp[c] = np.mean(scores)
imp = pd.Series(imp).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(imp.index, imp.values, color=COLORS["casual"], height=0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Increase in RMSE when shuffled (rides/day)")
ax.set_title(f"Feature importance: {best_name}")
plt.tight_layout(); plt.savefig("figures/09_feature_importance.png", bbox_inches="tight"); plt.show()

final_model = make_models()[best_name]
final_model.fit(X_all, y_all)

future = pd.date_range("2027-01-01", "2027-12-31", freq="D")
fc = pd.DataFrame({"casual_pred": final_model.predict(make_features(future))}, index=future)

# total riders forecast for the repair window: same model setup trained on member rides
member_model = make_models()[best_name]
member_model.fit(X_all, daily["member_rides"])
fc["member_pred"] = member_model.predict(make_features(future))
fc["total_pred"] = fc["casual_pred"] + fc["member_pred"]

weekly = fc.resample("W-SAT").agg(["sum", "count"])                # weeks run Sunday-Saturday
weekly = weekly.loc[weekly[("casual_pred", "count")] == 7].xs("sum", axis=1, level=1)  # full weeks only
weekly.index = weekly.index - pd.Timedelta(days=6)
weekly.index.name = "week_starting"

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(weekly.index, weekly["casual_pred"], color=COLORS["casual"], label="Casual (forecast)")
ax.plot(weekly.index, weekly["member_pred"], color=COLORS["member"], label="Member (forecast)")
top8 = weekly["casual_pred"].nlargest(8).index
ax.axvspan(top8.min(), top8.max() + pd.Timedelta(days=6), color=COLORS["casual"], alpha=0.08)
ax.text(top8.min(), ax.get_ylim()[0], " peak casual window", color="#52514e", va="bottom")
ax.yaxis.set_major_formatter(thousands)
ax.set_ylabel("Rides per week")
ax.set_title("2027 forecast: weekly rides")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("figures/10_forecast_2027.png", bbox_inches="tight"); plt.show()

print("Top 8 weeks for casual-rider ads in 2027:")
weekly.nlargest(8, "casual_pred")[["casual_pred", "total_pred"]].sort_index().round().astype(int)

print("Quietest 8 weeks for all riders (repair / maintenance window):")
weekly.nsmallest(8, "total_pred")[["total_pred", "casual_pred"]].sort_index().round().astype(int)

# Quietest hours overall, for same-day repair scheduling
hour_all = df.groupby("hour").size() / df["date"].nunique()
print("Average rides per hour across all riders, quietest 6 hours:")
hour_all.sort_values().head(6).round()

os.makedirs("outputs", exist_ok=True)
monthly.to_csv("outputs/monthly_rides_by_group.csv")
top_casual.to_csv("outputs/top_casual_stations.csv")
test_results.to_csv("outputs/model_test_results_2026.csv")
weekly.round().to_csv("outputs/forecast_2027_weekly.csv")
print("Saved tables to outputs/ and charts to figures/")
