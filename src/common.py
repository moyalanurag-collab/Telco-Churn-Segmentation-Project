"""Shared paths and helpers."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
PBI = ROOT / "outputs" / "powerbi"
ORANGE = ROOT / "outputs" / "orange"
for d in (PROC, FIG, PBI, ORANGE):
    d.mkdir(parents=True, exist_ok=True)

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

PALETTE = {"Churn": "#d1495b", "Stay": "#2e86ab", "grey": "#8d99ae"}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False})


def load_clean() -> pd.DataFrame:
    """Load raw data and apply the documented cleaning rules."""
    df = pd.read_csv(RAW)
    # TotalCharges has 11 blank strings: all are brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # collapse "No internet service"/"No phone service" into "No" for the add-on flags
    for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    return df


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
