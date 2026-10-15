"""Step 1 - data quality checks and exploratory analysis of churn drivers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from common import load_clean, save, PROC, PALETTE, ROOT

df = load_clean()
df.to_csv(PROC / "telco_clean.csv", index=False)
lines = []
log = lambda s="": (print(s), lines.append(s))

log("=== DATA QUALITY ===")
log(f"Rows: {len(df)}  Columns: {df.shape[1]}")
log(f"Duplicate customerIDs: {df.customerID.duplicated().sum()}")
log(f"Missing values after cleaning: {int(df.isna().sum().sum())}")
log("Raw TotalCharges had 11 blank values; all belong to tenure = 0 customers -> set to 0.")
overall = df.ChurnFlag.mean()
log(f"\nOverall churn rate: {overall:.1%} ({df.ChurnFlag.sum()} of {len(df)})")
log(f"Monthly revenue lost to churners: ${df.loc[df.ChurnFlag==1,'MonthlyCharges'].sum():,.0f} "
    f"of ${df.MonthlyCharges.sum():,.0f} ({df.loc[df.ChurnFlag==1,'MonthlyCharges'].sum()/df.MonthlyCharges.sum():.1%})")

# ---- categorical drivers: churn rate + chi-square -------------------------------------------
cats = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
rows = []
for c in cats:
    ct = pd.crosstab(df[c], df.ChurnFlag)
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    rates = df.groupby(c).ChurnFlag.mean()
    rows.append({"feature": c, "cramers_v": cramers_v, "p_value": p,
                 "highest_churn_level": rates.idxmax(), "highest_rate": rates.max(),
                 "lowest_churn_level": rates.idxmin(), "lowest_rate": rates.min()})
assoc = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
assoc.to_csv(PROC / "categorical_churn_association.csv", index=False)
log("\n=== CATEGORICAL DRIVERS (Cramer's V, chi-square) ===")
log(assoc.round(3).to_string(index=False))

# ---- numeric drivers ------------------------------------------------------------------------
log("\n=== NUMERIC DRIVERS (Mann-Whitney U) ===")
for c in ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]:
    a, b = df.loc[df.ChurnFlag == 1, c], df.loc[df.ChurnFlag == 0, c]
    _, p = mannwhitneyu(a, b)
    log(f"{c:15s} churners median={a.median():8.1f}  stayers median={b.median():8.1f}  p={p:.2e}")

# ---- figures --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
vc = df.Churn.value_counts()
ax.bar(vc.index, vc.values, color=[PALETTE["Stay"], PALETTE["Churn"]])
for i, v in enumerate(vc.values):
    ax.text(i, v + 50, f"{v:,}\n({v/len(df):.1%})", ha="center")
ax.set_title("Customer churn split"); ax.set_ylabel("Customers"); ax.set_ylim(0, vc.max() * 1.18)
save(fig, "01_churn_split.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                                "OnlineSecurity", "SeniorCitizen"]):
    r = df.groupby(c).ChurnFlag.mean().sort_values(ascending=False) * 100
    sns.barplot(x=r.values, y=r.index, ax=ax, color=PALETTE["Churn"])
    ax.axvline(overall * 100, ls="--", color="black", lw=1)
    for i, v in enumerate(r.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center")
    ax.set_title(f"Churn rate by {c}"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
    ax.set_xlim(0, r.max() * 1.18)
fig.suptitle(f"Churn rate by key attributes (dashed line = overall {overall:.1%})", fontweight="bold")
save(fig, "02_churn_by_category.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, c in zip(axes, ["tenure", "MonthlyCharges", "NumServices"]):
    for flag, lab in [(0, "Stay"), (1, "Churn")]:
        sns.kdeplot(df.loc[df.ChurnFlag == flag, c], ax=ax, fill=True, alpha=.35,
                    color=PALETTE[lab], label=lab, common_norm=False, bw_adjust=1.0 if c != "NumServices" else 1.5)
    ax.set_title(f"{c} distribution by churn"); ax.legend()
save(fig, "03_numeric_distributions.png")

df["TenureBand"] = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"])
fig, ax = plt.subplots(figsize=(7, 4))
r = df.groupby("TenureBand", observed=True).ChurnFlag.mean() * 100
ax.plot(r.index.astype(str), r.values, marker="o", color=PALETTE["Churn"], lw=2.5)
for x, v in zip(r.index.astype(str), r.values):
    ax.text(x, v + 1.5, f"{v:.0f}%", ha="center")
ax.set_ylim(0, r.max() * 1.2); ax.set_title("Churn risk collapses with tenure"); ax.set_ylabel("Churn rate (%)")
save(fig, "04_churn_by_tenure.png")

fig, ax = plt.subplots(figsize=(8, 5))
piv = df.pivot_table(index="Contract", columns="InternetService", values="ChurnFlag", aggfunc="mean") * 100
sns.heatmap(piv, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "Churn rate (%)"}, ax=ax)
ax.set_title("Churn rate (%): contract type x internet service")
save(fig, "05_contract_internet_heatmap.png")

num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "ChurnFlag"]]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation of numeric features")
save(fig, "06_correlation.png")

(ROOT / "outputs" / "eda_log.txt").write_text("\n".join(lines), encoding="utf-8")
