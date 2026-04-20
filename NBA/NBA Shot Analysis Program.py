# ---- Header -----

# Developer - Robert Hudson

# rhudsonprojects@outlook.com
# hudsonprojects78@gmail.com

# When I wrote this - only god and I knew how it worked.
# Now only god knows.


# PURPOSE:

# to perform analysis on NBA season 2005 - 2006 shot data

# creating two analysis points:
# attempting to model wins via total team EPS
# statistics showing EPS by shot area on court

# -------------------------------------------------------------------------------- #

# ----  Dependencies ---- 
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import statsmodels.api as sm
import scipy as stats



# ---- Parameters ----
OUTPUT_DIR = "./analysis"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- Ingest System ----
#script_dir = os.path.dirname(os.path.abspath(__file__)) #using same file dir as the scripting
#local_csvs = [f for f in os.listdir(script_dir) if f.lower().endswith(".csv")]

#csv_file = os.path.join(script_dir, local_csvs[0]) #R function I use to join basename and the folder path
#shots_data = pd.read_csv(csv_file)

# changed to direct pathing to prevent the csv search from returning two files and erroring out.
shots_data = pd.read_csv(r"C:\Users\Rober\Desktop\Coding Projects\Work\NBA Shot Analysis QMB 3311\NBA\nbaShots05_06.csv")

# ---- Manual Record Dataframe ----
# created df for team records
# changing ingest away from working directory. simpler for the two files.
records = pd.read_csv(r"C:\Users\Rober\Desktop\Coding Projects\Work\NBA Shot Analysis QMB 3311\NBA\nbaRecords05_06.csv")

print(records.to_string(index=False),'\n')

# ---- Data Cleaning ----
#converting to types
# date, y-m-d, numeric 
#adding shot point values

shots_data["GAME_DATE"] = pd.to_datetime(shots_data["GAME_DATE"], format="%Y%m%d")

for col in ["SHOT_MADE_FLAG", "SHOT_ATTEMPTED_FLAG", "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"]:
    if col in shots_data.columns:
        shots_data[col] = pd.to_numeric(shots_data[col])

shots_data["POINT_VALUE"] = shots_data["SHOT_TYPE"].apply(
    lambda x: 3 if "3pt" in str(x).lower() else 2 # ID if 
)


# ---- Summary Statistics, Ex Point p shot ----

zone_eps = (
    shots_data.groupby("SHOT_ZONE_BASIC")
    .agg(
        attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
        makes=("SHOT_MADE_FLAG", "sum"),
        avg_point_value=("POINT_VALUE", "mean"),
    )
    .reset_index()
)

zone_eps["fg_pct"] = zone_eps["makes"] / zone_eps["attempts"]
zone_eps["eps"] = zone_eps["fg_pct"] * zone_eps["avg_point_value"]
zone_eps = zone_eps.sort_values("eps", ascending=False)

print(zone_eps.to_string(index=False, float_format="%.3f"),'\n') # summary of the EPS calculated



# ---- Regressing, predicted wins from aggregated team eps ----
# comparing to true record

team_season_eps = (
    shots_data.groupby("TEAM_NAME")
    .agg(
        attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
        makes=("SHOT_MADE_FLAG", "sum"),
        avg_point_value=("POINT_VALUE", "mean"),
        avg_distance=("SHOT_DISTANCE", "mean"),
    )
    .reset_index()
)
# team specific aggregated eps via fg%*avg point value of shots
team_season_eps["fg_pct"] = team_season_eps["makes"] / team_season_eps["attempts"]
team_season_eps["eps"] = team_season_eps["fg_pct"] * team_season_eps["avg_point_value"]


# ---- Compare aggregated eps to record wins. ----

regression_df = team_season_eps.merge(records, on="TEAM_NAME", how="inner")

print(
    regression_df[["TEAM_NAME", "Wins", "Losses", "fg_pct", "eps", "attempts", "avg_distance"]]
    .sort_values("Wins", ascending=False)
    .to_string(index=False, float_format="%.3f")
)


# ---- Regression Modeling ----

X = sm.add_constant(regression_df["eps"])
y = regression_df["Wins"]
regression_model = sm.OLS(y, X).fit()

print(regression_model.summary())


# ---- teams performance vs predicted value ----

regression_df["predicted_wins"] = regression_model.predict(X)
regression_df["residual"] = regression_df["Wins"] - regression_df["predicted_wins"]
regression_df = regression_df.sort_values("residual")



# ---- Zone EPS chart ----
# tracking the expected points per shot y zone
Spot_eps_plot, ax = plt.subplots(figsize=(9, 5))

#color scaling by the range of the data 
colors = plt.cm.RdYlGn(
    (zone_eps["eps"] - zone_eps["eps"].min())
    / (zone_eps["eps"].max() - zone_eps["eps"].min())
)
bars = ax.barh(zone_eps["SHOT_ZONE_BASIC"], zone_eps["eps"], color=colors)
ax.set_xlabel("Expected Points per Shot")
ax.set_title(f"Expected Points per Shot by Zone")

# p[rinting the eps values for each zone
for bar, val in zip(bars, zone_eps["eps"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
Spot_eps_plot.savefig(f"{OUTPUT_DIR}/zone_eps.png")
plt.close(Spot_eps_plot)


# ---- Regression Chart ----
#pulling vals for chart
slope = regression_model.params["eps"] 
r2 = regression_model.rsquared
pval = regression_model.pvalues["eps"]

reg_plot, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    regression_df["eps"],
    regression_df["Wins"],
    s =regression_df["attempts"] / 40, #Scaling points
    alpha=0.7,# trans
    c= regression_df["Wins"], cmap = "RdYlGn", edgecolors= "black",
    linewidth= 0.5,
    zorder = 3,
)


x_line = np.linspace(regression_df["eps"].min() - 0.01, regression_df["eps"].max() + 0.01, 100)
y_line = regression_model.params["const"] + slope * x_line
ax.plot(x_line, y_line, color="red", linewidth=2.5, linestyle="--")

# Team names over their respective points
for _, row in regression_df.iterrows():
    ax.annotate(
        row["TEAM_NAME"],
        (row["eps"], row["Wins"]),
        fontsize=7,
        ha="center",
        va="bottom",
        xytext=(0, 6),
        textcoords="offset points",
    )

ax.set_xlabel("Expected Points per Shot", fontsize=11)
ax.set_ylabel("Wins", fontsize=11)
ax.set_title(f"Team Wins vs Shot Efficiency", fontsize=13)
plt.tight_layout()
reg_plot.savefig(f"{OUTPUT_DIR}/regression.png")
plt.close(reg_plot)

