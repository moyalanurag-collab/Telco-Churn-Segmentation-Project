"""Step 2 - RFM-style customer segmentation with K-Means.

The Telco data is a snapshot (no transaction history), so classic RFM is adapted:
  Recency   -> Tenure          months as a customer (relationship length)
  Frequency -> Service depth   number of services/products held
  Monetary  -> MonthlyCharges  current monthly revenue
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import save, PROC, PALETTE, ROOT

df = pd.read_csv(PROC / "telco_clean.csv")
feats = ["tenure", "NumServices", "MonthlyCharges"]
X = StandardScaler().fit_transform(df[feats])

# ---- choose k ---------------------------------------------------------------------------------
res = []
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    res.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_)})
sel = pd.DataFrame(res)
sel.to_csv(PROC / "kmeans_selection.csv", index=False)
print(sel.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(sel.k, sel.inertia, marker="o", color=PALETTE["Stay"]); axes[0].set_title("Elbow (inertia)")
axes[1].plot(sel.k, sel.silhouette, marker="o", color=PALETTE["Churn"]); axes[1].set_title("Silhouette score")
for a in axes: a.set_xlabel("k (number of clusters)")
save(fig, "07_kmeans_k_selection.png")

K = 5  # chosen for business usability: enough resolution for distinct actions, still manageable
km = KMeans(K, n_init=50, random_state=42).fit(X)
df["cluster"] = km.labels_
print(f"\nChosen k={K}: silhouette={silhouette_score(X, km.labels_):.3f}")

prof = df.groupby("cluster").agg(
    customers=("customerID", "count"), tenure=("tenure", "mean"), services=("NumServices", "mean"),
    monthly=("MonthlyCharges", "mean"), churn=("ChurnFlag", "mean"),
    pct_m2m=("Contract", lambda s: (s == "Month-to-month").mean()),
    pct_fiber=("InternetService", lambda s: (s == "Fiber optic").mean()),
    pct_nointernet=("InternetService", lambda s: (s == "No").mean()))
print(prof.round(2))
prof.to_csv(PROC / "cluster_profile_raw.csv")
df.to_csv(PROC / "telco_clustered_unnamed.csv", index=False)
np.save(PROC / "X_scaled.npy", X)
