# Power BI dashboard - quick build guide (about 30-45 minutes)

*Project: Anurag Moyal - Telco Churn & Segmentation*

Everything is already prepared in `outputs/powerbi/`. To see the finished design first, double-click
`outputs/Churn_Dashboard.html` (an interactive preview with the same pages, filters and numbers).

| File in `outputs/powerbi/` | Use |
|---|---|
| `fact_customers.csv` | main table (1 row per customer, already includes segment, risk band, churn probability, sort-order columns) |
| `dim_segments.csv` | segment table with priority, programme and recommended action |
| `retention_target_list_top500.csv` | top 500 active customers to contact |
| `dim_risk_bands.csv`, `agg_segment_risk.csv` | optional summary tables |
| `Telco_Churn_Theme.json` | colours/fonts: View > Themes > Browse for themes |
| `DAX_measures.txt` | every measure, ready to copy and paste |

## Step 1 - Load (5 min)
Home > Get data > Text/CSV, load `fact_customers.csv`, `dim_segments.csv`, `retention_target_list_top500.csv`.
Rename the tables to `Customers`, `Segments`, `TargetList`. In the preview dialog just click **Load**.

## Step 2 - Model (3 min)
Model view: drag `Customers[Segment]` onto `Segments[Segment]` (many-to-one, single direction).
Data view > select column `risk_band` > Column tools > **Sort by column** > `RiskOrder`. Do the same for
`TenureBand` -> `TenureOrder`, and `ChargeBand` -> `ChargeOrder`.

## Step 3 - Theme and measures (5 min)
View > Themes > Browse for themes > pick `Telco_Churn_Theme.json`.
Open `DAX_measures.txt` and create each line as a new measure (Home > New measure).

## Step 4 - Build 3 pages
**Page 1 "Overview"** - title: *Two fibre segments drive 71% of churn*
* 6 cards: `Customers`, `Churn Rate` (format %), `Revenue Lost per Month` ($), `Revenue Lost %`, `High-Risk Active`, `Annual Revenue at Risk`
* Clustered bar: Y `Segment`, X `Churn Rate`, with a constant line at 26.5% (Analytics pane)
* Column chart: X `TenureBand`, Y `Churn Rate`
* Slicers: `Contract`, `InternetService`, `PaymentMethod`

**Page 2 "Segments"** - title: *Where the money is lost*
* Scatter: X `Avg Tenure`, Y `Avg Monthly Charge`, Size `Customers`, Legend `Segment`
* Matrix: rows `Segment`; values `Customers`, `Avg Monthly Charge`, `Churn Rate`, `Revenue Lost per Month`, `Share of All Churn` (red conditional formatting on `Churn Rate`)
* Table from `Segments`: `priority`, `Segment`, `recommended_programme`, `recommended_action` (the "who to target" answer)

**Page 3 "Who to call"** - title: *Prioritised retention list*
* Stacked column: X `risk_band`, Y `Customers`
* Matrix: rows `Segment`, columns `risk_band`, values `Customers` (heat colouring)
* Table from `TargetList` sorted by `expected_annual_revenue_at_risk` descending; slicers `Segment`, `risk_band`

## Step 5 - Finish
File > Save as `Telco_Churn_Dashboard.pbix`. Export PDF (File > Export) and screenshot each page for your portfolio / CV.

## Notes for explaining it
* Risk bands: Low <15%, Medium 15-40%, High 40-65%, Critical >65% predicted churn probability (out-of-fold model scores, so honest).
* `expected_annual_revenue_at_risk` = churn probability x monthly charge x 12.
* Campaign economics ($40 per contact, 25% acceptance, 12 months saved) are assumptions, not company facts.
