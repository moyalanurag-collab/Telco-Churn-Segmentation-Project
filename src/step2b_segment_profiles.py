"""Step 2b - name the K-Means clusters, profile each segment, and draw the segment figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from common import save, PROC

df = pd.read_csv(PROC / "telco_clustered_unnamed.csv")
X = np.load(PROC / "X_scaled.npy")
prof = pd.read_csv(PROC / "cluster_profile_raw.csv", index_col="cluster")


def name_segment(r):
    """Deterministic naming rules on the cluster centroids (no manual labelling)."""
    if r.tenure > 45 and r.monthly > 70:
        return "Premium Loyalists"
    if r.tenure > 45:
        return "Loyal Basic Savers"
    if r.monthly < 40:
        return "New Budget Starters"
    if r.tenure > 18:
        return "Mid-Tenure Fiber At-Risk"
    return "New Fiber Switchers"


names = prof.apply(name_segment, axis=1)
assert names.is_unique, names
df["Segment"] = df["cluster"].map(names)
order = df.groupby("Segment").ChurnFlag.mean().sort_values(ascending=False).index.tolist()

seg = df.groupby("Segment").agg(
    customers=("customerID", "count"), avg_tenure_months=("tenure", "mean"),
    avg_services=("NumServices", "mean"), avg_monthly_charge=("MonthlyCharges", "mean"),
    monthly_revenue=("MonthlyCharges", "sum"), churn_rate=("ChurnFlag", "mean"),
    churned=("ChurnFlag", "sum"),
    pct_month_to_month=("Contract", lambda s: (s == "Month-to-month").mean()),
    pct_fiber=("InternetService", lambda s: (s == "Fiber optic").mean()),
    pct_electronic_check=("PaymentMethod", lambda s: (s == "Electronic check").mean()),
    pct_tech_support=("TechSupport", lambda s: (s == "Yes").mean())).loc[order]
seg["monthly_revenue_lost"] = df[df.ChurnFlag == 1].groupby("Segment").MonthlyCharges.sum()
seg["share_of_customers"] = seg.customers / len(df)
seg["share_of_churn"] = seg.churned / seg.churned.sum()
seg["share_of_revenue_lost"] = seg.monthly_revenue_lost / seg.monthly_revenue_lost.sum()
seg.to_csv(PROC / "segment_summary.csv")
pd.set_option("display.width", 220)
print(seg.round(2).T.to_string())
df.drop(columns="cluster").to_csv(PROC / "telco_segmented.csv", index=False)

cols = dict(zip(order, ["#d1495b", "#ee964b", "#f4d35e", "#2e86ab", "#4c956c"]))
pca = PCA(2, random_state=42).fit(X)
Z = pca.transform(X)
fig, ax = plt.subplots(figsize=(8, 6))
for s in order:
    m = (df.Segment == s).values
    ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=.5, color=cols[s], label=f"{s} (n={m.sum():,})")
ax.set_title("Customer segments (PCA of tenure, services, monthly charge)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.legend(markerscale=3, fontsize=8)
save(fig, "08_segments_pca.png")

base = df[["tenure", "NumServices", "MonthlyCharges"]]
z = (seg[["avg_tenure_months", "avg_services", "avg_monthly_charge"]].values - base.mean().values) / base.std().values
z = pd.DataFrame(z, index=seg.index, columns=["Tenure (R)", "Services (F)", "Monthly charge (M)"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(z, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "std. dev. vs average"})
ax.set_title("RFM-style profile of each segment"); ax.set_ylabel("")
save(fig, "09_segment_rfm_heatmap.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
spec = [("churn_rate", 100, "Churn rate (%)", "{:.0f}%"), ("share_of_customers", 100, "Share of customers (%)", "{:.0f}%"),
        ("monthly_revenue_lost", 1, "Monthly revenue lost to churn ($)", "${:,.0f}")]
for i, (a, (col, mult, title, fmt)) in enumerate(zip(axes, spec)):
    vals = seg[col] * mult
    a.barh(seg.index, vals, color=[cols[s] for s in seg.index])
    for j, v in enumerate(vals):
        a.text(v * 1.01, j, fmt.format(v), va="center", fontsize=9)
    a.set_title(title); a.margins(x=.22)
    if i:
        a.set_yticklabels([])
axes[0].axvline(df.ChurnFlag.mean() * 100, ls="--", color="black")
save(fig, "10_segment_churn_overview.png")
