"""Step 4 - export Power BI star-schema tables, an Orange Data Mining file."""
import json
import pandas as pd
from common import PROC, PBI, ORANGE

df = pd.read_csv(PROC / "telco_scored.csv")
ms = json.load(open(PROC / "model_summary.json"))
thr = ms["threshold"]

df["TenureBand"] = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"])
df["ChargeBand"] = pd.cut(df.MonthlyCharges, [0, 35, 70, 90, 200], labels=["<$35", "$35-70", "$70-90", "$90+"])
df["ChurnLabel"] = df.Churn
df["risk_band"] = pd.Categorical(df.risk_band, ["Low", "Medium", "High", "Critical"], ordered=True)

# ---- fact table: one row per customer -----------------------------------------------------------
fact_cols = ["customerID", "Segment", "risk_band", "churn_probability", "predicted_churn", "ChurnFlag", "ChurnLabel",
             "tenure", "TenureBand", "MonthlyCharges", "ChargeBand", "TotalCharges", "annual_revenue",
             "expected_annual_revenue_at_risk", "NumServices", "Contract", "InternetService", "PaymentMethod",
             "PaperlessBilling", "TechSupport", "OnlineSecurity", "SeniorCitizen", "Partner", "Dependents", "gender"]
df["RiskOrder"] = df.risk_band.map({"Low": 1, "Medium": 2, "High": 3, "Critical": 4})
df["TenureOrder"] = df.TenureBand.map({"0-6m": 1, "7-12m": 2, "13-24m": 3, "25-48m": 4, "49-72m": 5})
df["ChargeOrder"] = df.ChargeBand.map({"<$35": 1, "$35-70": 2, "$70-90": 3, "$90+": 4})
fact_cols += ["RiskOrder", "TenureOrder", "ChargeOrder"]
fact = df[fact_cols].copy()
fact["churn_probability"] = fact.churn_probability.round(4)
fact["annual_revenue"] = fact.annual_revenue.round(2)
fact["expected_annual_revenue_at_risk"] = fact.expected_annual_revenue_at_risk.round(2)
fact.to_csv(PBI / "fact_customers.csv", index=False)

# ---- dimension: segment playbook (the "who to target" table) ---------------------------------------
seg = pd.read_csv(PROC / "segment_summary.csv")
play = {
    "New Fiber Switchers": ("1 - Critical", "Early-life onboarding",
                            "Month-to-month fiber customers in their first year: 47% churn. Run a 90-day onboarding journey, "
                            "offer a 12-month contract with a bill credit, add free Online Security / Tech Support trial, "
                            "and move them off electronic check to auto-pay."),
    "Mid-Tenure Fiber At-Risk": ("2 - Critical", "Contract upgrade + service bundling",
                                 "High-value (~$87/mo) fiber customers, 75% month-to-month, only 37% have Tech Support. "
                                 "Offer price-lock 1-2 year contracts and a bundled security/support add-on; proactive "
                                 "service-quality check for fiber faults."),
    "New Budget Starters": ("3 - Nurture", "Low-cost digital nurture",
                            "Low-spend DSL/phone-only newcomers (21% churn). Automated emails/SMS, upsell one add-on, "
                            "avoid expensive discounts - limited revenue at stake."),
    "Premium Loyalists": ("4 - Protect", "Loyalty and VIP care",
                          "Largest revenue pool ($163K/mo) with 13% churn. Protect with loyalty rewards, priority support "
                          "and renewal reminders; watch the month-to-month minority."),
    "Loyal Basic Savers": ("5 - Maintain", "Maintain",
                           "Very sticky (4% churn). No retention spend; occasional cross-sell only."),
}
seg["priority"] = seg.Segment.map(lambda s: play[s][0])
seg["recommended_programme"] = seg.Segment.map(lambda s: play[s][1])
seg["recommended_action"] = seg.Segment.map(lambda s: play[s][2])
for c in seg.select_dtypes("float").columns:
    seg[c] = seg[c].round(4)
seg.to_csv(PBI / "dim_segments.csv", index=False)

# ---- risk-band table ---------------------------------------------------------------------------------
rb = df.groupby("risk_band", observed=True).agg(
    customers=("customerID", "count"), actual_churn_rate=("ChurnFlag", "mean"), avg_predicted=("churn_probability", "mean"),
    monthly_revenue=("MonthlyCharges", "sum"), expected_annual_revenue_at_risk=("expected_annual_revenue_at_risk", "sum")).reset_index()
rb.to_csv(PBI / "dim_risk_bands.csv", index=False)

# ---- segment x risk matrix ------------------------------------------------------------------------------
sr = df.groupby(["Segment", "risk_band"], observed=True).agg(
    customers=("customerID", "count"), actual_churn_rate=("ChurnFlag", "mean"),
    expected_annual_revenue_at_risk=("expected_annual_revenue_at_risk", "sum")).reset_index()
sr.to_csv(PBI / "agg_segment_risk.csv", index=False)

# ---- retention target list (top 500 by expected revenue at risk, not yet churned) ------------------------
tl = df[df.ChurnFlag == 0].sort_values("expected_annual_revenue_at_risk", ascending=False).head(500)[
    ["customerID", "Segment", "risk_band", "churn_probability", "MonthlyCharges", "tenure", "Contract", "InternetService",
     "PaymentMethod", "TechSupport", "expected_annual_revenue_at_risk"]]
tl.to_csv(PBI / "retention_target_list_top500.csv", index=False)

# ---- Orange Data Mining .tab (3 header rows: names / types / roles) ---------------------------------------
oc = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
      "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
      "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "NumServices", "Segment", "Churn"]
o = df[oc]
header_types = ["continuous" if pd.api.types.is_numeric_dtype(o[c]) else "discrete" for c in oc]
flags = ["meta" if c == "Segment" else ("class" if c == "Churn" else "") for c in oc]
with open(ORANGE / "telco_churn_orange.tab", "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(oc) + "\n" + "\t".join(header_types) + "\n" + "\t".join(flags) + "\n")
    o.to_csv(f, sep="\t", header=False, index=False)

# ---- numbers quoted in the report ---------------------------------------------------------------------------
hi = df[df.churn_probability >= thr]
print(f"Customers >= threshold {thr:.2f}: {len(hi)} ({len(hi)/len(df):.1%}); actual churn in that group {hi.ChurnFlag.mean():.1%}")
print(f"Active (not churned) customers flagged: {(hi.ChurnFlag==0).sum()}, their annual revenue ${hi.loc[hi.ChurnFlag==0,'annual_revenue'].sum():,.0f}")
print(f"Total expected annual revenue at risk (all): ${df.expected_annual_revenue_at_risk.sum():,.0f}")
print(rb.round(3).to_string(index=False))
print(sr.round(2).to_string(index=False))
# segments whose churn rate is more than 1.4x overall = "high-risk"
print(seg[["Segment", "customers", "churn_rate", "share_of_churn", "share_of_revenue_lost"]].round(3).to_string(index=False))
hr = seg[seg.churn_rate > 1.4 * df.ChurnFlag.mean()]
print("High-risk segments:", hr.Segment.tolist(), f"-> {hr.customers.sum()} customers ({hr.customers.sum()/len(df):.0%}), "
      f"{hr.share_of_churn.sum():.0%} of all churn, {hr.share_of_revenue_lost.sum():.0%} of lost revenue")
