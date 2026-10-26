"""Step 3 - churn prediction: model comparison, evaluation, drivers and retention economics."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss)
from xgboost import XGBClassifier
from common import save, PROC, PALETTE

RS = 42
df = pd.read_csv(PROC / "telco_segmented.csv")
num = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
cat = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
       "Contract", "PaperlessBilling", "PaymentMethod"]
X, y = df[num + cat], df.ChurnFlag
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)


def prep(scale=True):
    return ColumnTransformer([("n", StandardScaler() if scale else "passthrough", num),
                              ("c", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])


models = {
    "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestClassifier(
        500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RS))]),
    "Gradient Boosting": Pipeline([("p", prep(False)), ("m", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RS))]),
    "XGBoost": Pipeline([("p", prep(False)), ("m", XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), eval_metric="logloss", random_state=RS, n_jobs=-1))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
rows, fitted = [], {}
for name, pipe in models.items():
    s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["roc_auc", "average_precision"])
    pipe.fit(Xtr, ytr)
    fitted[name] = pipe
    p = pipe.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    rows.append({"model": name, "cv_auc_mean": s["test_roc_auc"].mean(), "cv_auc_std": s["test_roc_auc"].std(),
                 "test_auc": roc_auc_score(yte, p), "test_pr_auc": average_precision_score(yte, p),
                 "accuracy@0.5": accuracy_score(yte, pred), "precision@0.5": precision_score(yte, pred),
                 "recall@0.5": recall_score(yte, pred), "f1@0.5": f1_score(yte, pred), "brier": brier_score_loss(yte, p)})
comp = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
comp.to_csv(PROC / "model_comparison.csv", index=False)
pd.set_option("display.width", 220)
print(comp.round(3).to_string(index=False))
best = comp.iloc[0].model
base_rate = yte.mean()
print(f"Best by CV AUC: {best}.  Test base churn rate {base_rate:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, pipe in fitted.items():
    p = pipe.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p)
    axes[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(title="ROC curves (hold-out test set)", xlabel="False positive rate", ylabel="True positive rate")
axes[0].legend(loc="lower right")
axes[1].axhline(base_rate, color="k", ls="--", lw=1)
axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
axes[1].legend()
save(fig, "11_roc_pr_curves.png")

# out-of-fold risk score for EVERY customer (used by the dashboard)
champ = models[best]
oof = cross_val_predict(champ, X, y, cv=cv, method="predict_proba")[:, 1]
df["churn_probability"] = oof
print("Out-of-fold AUC, all customers:", round(roc_auc_score(y, oof), 3))

# ---- threshold chosen on TRAIN out-of-fold scores using retention economics --------------------
# Editable business assumptions: $40 per outreach (agent time + incentive credit), 25% of contacted
# would-be churners accept, and a saved customer stays 12 more months at their current monthly charge.
CONTACT_COST, OFFER_SUCCESS, MONTHS_SAVED = 40.0, 0.25, 12


def net_benefit(prob, truth, mrr, thr):
    t = prob >= thr
    return (truth[t] * mrr[t] * MONTHS_SAVED * OFFER_SUCCESS).sum() - CONTACT_COST * t.sum()


oof_tr = cross_val_predict(champ, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
mrr_tr, mrr_te = Xtr.MonthlyCharges.values, Xte.MonthlyCharges.values
grid = np.arange(0.05, 0.95, 0.01)
nb_tr = [net_benefit(oof_tr, ytr.values, mrr_tr, t) for t in grid]
thr = float(grid[int(np.argmax(nb_tr))])
p_te = fitted[best].predict_proba(Xte)[:, 1]
pred = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"Threshold {thr:.2f}: precision {precision_score(yte, pred):.3f} recall {recall_score(yte, pred):.3f} "
      f"F1 {f1_score(yte, pred):.3f}  TP {tp} FP {fp} FN {fn} TN {tn}")
nb_te = [net_benefit(p_te, yte.values, mrr_te, t) for t in grid]
nb_model = net_benefit(p_te, yte.values, mrr_te, thr)
nb_all = net_benefit(np.ones(len(yte)), yte.values, mrr_te, 0)
print(f"Test net benefit: model ${nb_model:,.0f} vs contact-everyone ${nb_all:,.0f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid, np.array(nb_te) / len(yte), color=PALETTE["Stay"], lw=2, label="Test set")
ax.plot(grid, np.array(nb_tr) / len(ytr), color=PALETTE["grey"], lw=1.5, ls=":", label="Train (out-of-fold)")
ax.axvline(thr, color=PALETTE["Churn"], ls="--", label=f"Chosen threshold {thr:.2f}")
ax.axhline(nb_all / len(yte), color="black", lw=1, ls="--", label="Contact everyone (test)")
ax.set(xlabel="Probability threshold for retention outreach", ylabel="Net benefit per customer ($)",
       title=f"Retention campaign economics ({best})")
ax.legend(fontsize=8)
save(fig, "12_profit_curve.png")

fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred stay", "Pred churn"], yticklabels=["Actual stay", "Actual churn"])
ax.set_title(f"Confusion matrix @ threshold {thr:.2f}")
save(fig, "13_confusion_matrix.png")

o = np.argsort(-p_te)
cum = np.cumsum(yte.values[o]) / yte.sum()
frac = np.arange(1, len(o) + 1) / len(o)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(frac, cum, color=PALETTE["Churn"], lw=2, label="Model")
ax.plot([0, 1], [0, 1], "k--", label="Random")
for q in (0.2, 0.3):
    i = int(q * len(o))
    ax.scatter(q, cum[i], color="black", zorder=5)
    ax.annotate(f"top {int(q * 100)}% -> {cum[i]:.0%} of churners", (q, cum[i]), (q + .05, cum[i] - .12), fontsize=9)
ax.set(xlabel="Share of customers contacted (highest risk first)", ylabel="Share of churners captured", title="Cumulative gains")
ax.legend()
save(fig, "14_gains_curve.png")
top20, top30 = cum[int(.2 * len(o))], cum[int(.3 * len(o))]
print(f"Top 20% riskiest capture {top20:.1%} of churners (lift {top20 / .2:.2f}x); top 30%: {top30:.1%}")

fpv, mpv = calibration_curve(yte, p_te, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(mpv, fpv, marker="o", color=PALETTE["Stay"])
ax.plot([0, 1], [0, 1], "k--")
ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration (class weights inflate scores)")
save(fig, "15_calibration.png")

# ---- drivers --------------------------------------------------------------------------------------
lr = fitted["Logistic Regression"]
fn_ = lr.named_steps["p"].get_feature_names_out()
coef = pd.DataFrame({"feature": [n.split("__")[1] for n in fn_], "coef": lr.named_steps["m"].coef_[0]})
coef["odds_ratio"] = np.exp(coef.coef)
coef = coef.reindex(coef.coef.abs().sort_values(ascending=False).index)
coef.to_csv(PROC / "logistic_odds_ratios.csv", index=False)
t = coef.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(t.feature, t.coef, color=[PALETTE["Churn"] if c > 0 else PALETTE["Stay"] for c in t.coef])
ax.axvline(0, color="black", lw=1)
ax.set(title="Logistic regression drivers (red raises churn risk)", xlabel="Coefficient (log-odds, numerics standardised)")
save(fig, "16_logistic_drivers.png")

pi = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=15, random_state=RS)
imp = pd.DataFrame({"feature": Xte.columns, "auc_drop": pi.importances_mean, "sd": pi.importances_std}
                   ).sort_values("auc_drop", ascending=False)
imp.to_csv(PROC / "permutation_importance.csv", index=False)
t = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(t.feature, t.auc_drop, xerr=t.sd, color=PALETTE["Stay"])
ax.set(title=f"Permutation importance ({best})", xlabel="Drop in test AUC when feature is shuffled")
save(fig, "17_permutation_importance.png")
print("\nTop drivers:\n", imp.head(8).round(4).to_string(index=False))
print("\nTop odds ratios:\n", coef.head(10).round(2).to_string(index=False))

# ---- scored customer table ----------------------------------------------------------------------------
df["risk_band"] = pd.cut(df.churn_probability, [-.01, 0.15, 0.40, 0.65, 1.01], labels=["Low", "Medium", "High", "Critical"])
df["predicted_churn"] = (df.churn_probability >= thr).astype(int)
df["annual_revenue"] = df.MonthlyCharges * 12
df["expected_annual_revenue_at_risk"] = df.churn_probability.clip(0, 1) * df.annual_revenue
df.to_csv(PROC / "telco_scored.csv", index=False)
json.dump({"best_model": best, "threshold": thr, "oof_auc_all": float(roc_auc_score(y, oof)),
           "gains_top20": float(top20), "gains_top30": float(top30),
           "assumptions": {"contact_cost": CONTACT_COST, "offer_success": OFFER_SUCCESS, "months_saved": MONTHS_SAVED},
           "test_precision": float(precision_score(yte, pred)), "test_recall": float(recall_score(yte, pred)),
           "net_benefit_model": float(nb_model), "net_benefit_all": float(nb_all), "test_n": int(len(yte))},
          open(PROC / "model_summary.json", "w"), indent=2)
print(df.groupby("risk_band", observed=True).agg(n=("customerID", "count"), actual_churn=("ChurnFlag", "mean")).round(3))
