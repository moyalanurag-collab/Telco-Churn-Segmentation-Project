# Orange Data Mining - visual model comparison (about 15 minutes)

*Project: Anurag Moyal - Telco Churn & Segmentation*

Download Orange (free) from orangedatamining.com. The prepared file is `outputs/orange/telco_churn_orange.tab`:
`Churn` is already marked as the **class**, and `Segment` as a **meta** column (kept for display, not used for learning).

## Workflow (drag these widgets onto the canvas and connect left to right)
```
File -> Data Table                       (check 7,043 rows, 21 features)
File -> Rank                             (which features matter: Info gain / Chi2 / ReliefF)
File -> Select Columns? (optional) -----+
File ------------------------------------+-> Test and Score  <- Logistic Regression
                                                            <- Random Forest
                                                            <- Gradient Boosting
                                                            <- kNN (baseline)
                                                            <- Naive Bayes (baseline)
Test and Score -> ROC Analysis           (curves for all models)
Test and Score -> Confusion Matrix       (errors per model)
Logistic Regression -> Nomogram          (explainable churn drivers)
Random Forest -> Feature Importance / Explain model  (optional)
File -> Distributions / Box Plot / Mosaic Display    (EDA: Contract, InternetService vs Churn)
```
Settings: in **Test and Score** use *Cross validation, 5 folds, stratified*; set **Target class = Yes**.
Open **ROC Analysis** and tick *Show convex ROC hull*.

## What you should see (Python results for comparison)
Cross-validated AUC is about 0.845-0.849 for logistic regression, random forest, gradient boosting and XGBoost,
so the models are close; kNN and Naive Bayes are typically a bit lower. Top ranked features: **Contract, tenure,
InternetService, MonthlyCharges, TotalCharges**. If Orange shows the same ordering, you have confirmed the Python
findings with a second, independent tool - say that in the write-up.

## Save
File -> Save as `Telco_Churn_Orange_Workflow.ows`, and screenshot the ROC Analysis and Rank widgets for the report.
