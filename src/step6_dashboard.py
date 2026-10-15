"""Step 6 - build a self-contained interactive HTML dashboard (mirrors the Power BI design)."""
import json
import pandas as pd
from common import PBI, ROOT

df = pd.read_csv(PBI / "fact_customers.csv")
seg = pd.read_csv(PBI / "dim_segments.csv")
ms = json.load(open(ROOT / "data" / "processed" / "model_summary.json"))

cols = ["customerID", "Segment", "risk_band", "churn_probability", "predicted_churn", "ChurnFlag", "tenure", "TenureBand",
        "MonthlyCharges", "expected_annual_revenue_at_risk", "Contract", "InternetService", "PaymentMethod"]
rows = df[cols].values.tolist()
plays = {r.Segment: [r.priority, r.recommended_programme, r.recommended_action] for r in seg.itertuples()}

html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Telco Churn Dashboard</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--mut:#6b7385;--red:#d1495b;--blue:#2e86ab;--line:#e3e6ec}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e8ebf2;--mut:#98a1b5;--line:#2a313e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:20px 24px 4px}h1{margin:0;font-size:22px}.sub{color:var(--mut);margin-top:2px}
.filters{display:flex;flex-wrap:wrap;gap:10px;padding:12px 24px}
.filters label{font-size:12px;color:var(--mut);display:flex;flex-direction:column;gap:3px}
select{padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;padding:0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;padding:12px 24px 24px}
h2{font-size:14px;margin:0 0 10px}
.bar{display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;margin:5px 0;font-size:13px}
.track{background:var(--line);border-radius:4px;height:14px;position:relative}.fill{height:100%;border-radius:4px;background:var(--red)}
.ref{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}.wide{grid-column:1/-1}.tag{padding:2px 7px;border-radius:9px;color:#fff;font-size:11px}
.note{color:var(--mut);font-size:12px;padding:0 24px 24px}
.scroll{overflow-x:auto}
</style></head><body>
<header><h1>Telco Customer Churn &amp; Segmentation Dashboard</h1>
<div class="sub">Prepared by Anurag Moyal &middot; 7,043 customers &middot; model: __MODEL__ (AUC __AUC__)</div></header>
<div class="filters" id="filters"></div>
<div class="kpis" id="kpis"></div>
<div class="grid">
<div class="card"><h2>Churn rate by segment <span class="sub">(line = filtered average)</span></h2><div id="c_seg"></div></div>
<div class="card"><h2>Churn rate by tenure band</h2><div id="c_ten"></div></div>
<div class="card"><h2>Churn rate by contract</h2><div id="c_con"></div></div>
<div class="card"><h2>Churn rate by internet service</h2><div id="c_int"></div></div>
<div class="card"><h2>Churn rate by payment method</h2><div id="c_pay"></div></div>
<div class="card"><h2>Customers by model risk band <span class="sub">(actual churn rate shown)</span></h2><div id="c_risk"></div></div>
<div class="card wide scroll"><h2>Segment playbook - who to target</h2><table id="t_seg"></table></div>
<div class="card wide scroll"><h2>Top 25 active customers to contact (highest expected annual revenue at risk)</h2><table id="t_tgt"></table></div>
</div>
<div class="note">Risk bands: Low &lt;15%, Medium 15-40%, High 40-65%, Critical &gt;65% predicted churn probability. Expected annual revenue at risk = churn probability x 12 x monthly charge.
Campaign threshold __THR__ chosen from assumed economics ($__COST__ per contact, __SUC__% offer acceptance, __MON__ months saved). Predictions are out-of-fold.</div>
<script>
const D=__DATA__, PLAY=__PLAY__;
const F={Segment:"All",Contract:"All",InternetService:"All",PaymentMethod:"All"};
const uniq=i=>[...new Set(D.map(r=>r[i]))].sort();
const colOf={Segment:1,Contract:10,InternetService:11,PaymentMethod:12};
const fmt=n=>n.toLocaleString("en-US",{maximumFractionDigits:0});
const usd=n=>"$"+fmt(n);
const pct=x=>(x*100).toFixed(1)+"%";
function build(){const f=document.getElementById("filters");
 for(const k of Object.keys(F)){const l=document.createElement("label");l.textContent=k;const s=document.createElement("select");
  s.innerHTML="<option>All</option>"+uniq(colOf[k]).map(v=>`<option>${v}</option>`).join("");
  s.onchange=()=>{F[k]=s.value;draw()};l.appendChild(s);f.appendChild(l)}}
function rate(rs){return rs.length?rs.reduce((a,r)=>a+r[5],0)/rs.length:0}
function bars(id,groupIdx,order,rs,avg){const g={};rs.forEach(r=>(g[r[groupIdx]]??=[]).push(r));
 const keys=(order||Object.keys(g)).filter(k=>g[k]);const mx=Math.max(.01,...keys.map(k=>rate(g[k])),avg);
 document.getElementById(id).innerHTML=keys.map(k=>{const v=rate(g[k]);
  return `<div class="bar"><span>${k} <span class="sub">(${fmt(g[k].length)})</span></span><div class="track"><div class="fill" style="width:${v/mx*100}%"></div><div class="ref" style="left:${avg/mx*100}%"></div></div><b>${pct(v)}</b></div>`}).join("")}
function draw(){
 const rs=D.filter(r=>Object.keys(F).every(k=>F[k]=="All"||r[colOf[k]]==F[k]));
 const ch=rs.filter(r=>r[5]==1),act=rs.filter(r=>r[5]==0),flag=act.filter(r=>r[4]==1),avg=rate(rs);
 const k=[["Customers",fmt(rs.length)],["Churn rate",pct(avg)],["Revenue lost / month",usd(ch.reduce((a,r)=>a+r[8],0))],
  ["Revenue lost %",pct(rs.length?ch.reduce((a,r)=>a+r[8],0)/rs.reduce((a,r)=>a+r[8],0):0)],
  ["High-risk active customers",fmt(flag.length)],["Expected annual revenue at risk",usd(act.reduce((a,r)=>a+r[9],0))]];
 document.getElementById("kpis").innerHTML=k.map(([l,v])=>`<div class="card kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
 const segOrder=Object.keys(PLAY).sort((a,b)=>PLAY[a][0].localeCompare(PLAY[b][0]));
 bars("c_seg",1,segOrder,rs,avg);bars("c_ten",7,["0-6m","7-12m","13-24m","25-48m","49-72m"],rs,avg);
 bars("c_con",10,null,rs,avg);bars("c_int",11,null,rs,avg);bars("c_pay",12,null,rs,avg);
 const rc={Low:"#4c956c",Medium:"#f4d35e",High:"#ee964b",Critical:"#d1495b"};
 const g={};rs.forEach(r=>(g[r[2]]??=[]).push(r));const mx=Math.max(1,...Object.values(g).map(a=>a.length));
 document.getElementById("c_risk").innerHTML=["Low","Medium","High","Critical"].filter(b=>g[b]).map(b=>
  `<div class="bar"><span>${b}</span><div class="track"><div class="fill" style="width:${g[b].length/mx*100}%;background:${rc[b]}"></div></div><b>${fmt(g[b].length)} (${pct(rate(g[b]))})</b></div>`).join("");
 document.getElementById("t_seg").innerHTML="<tr><th>Priority</th><th>Segment</th><th>Customers</th><th>Churn</th><th>Rev lost/mo</th><th>Programme</th><th>Action</th></tr>"+
  segOrder.map(s=>{const a=rs.filter(r=>r[1]==s);if(!a.length)return"";return `<tr><td>${PLAY[s][0]}</td><td><b>${s}</b></td><td>${fmt(a.length)}</td><td>${pct(rate(a))}</td><td>${usd(a.filter(r=>r[5]==1).reduce((x,r)=>x+r[8],0))}</td><td>${PLAY[s][1]}</td><td>${PLAY[s][2]}</td></tr>`}).join("");
 const top=[...act].sort((a,b)=>b[9]-a[9]).slice(0,25);
 document.getElementById("t_tgt").innerHTML="<tr><th>Customer</th><th>Segment</th><th>Risk</th><th>Churn prob.</th><th>Monthly $</th><th>Tenure</th><th>Contract</th><th>Internet</th><th>Exp. annual rev at risk</th></tr>"+
  top.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td><span class="tag" style="background:${rc[r[2]]}">${r[2]}</span></td><td>${pct(r[3])}</td><td>${r[8].toFixed(2)}</td><td>${r[6]}m</td><td>${r[10]}</td><td>${r[11]}</td><td>${usd(r[9])}</td></tr>`).join("");
}
build();draw();
</script></body></html>"""

a = ms["assumptions"]
html = (html.replace("__DATA__", json.dumps([[r[0], r[1], r[2], round(r[3], 3), r[4], r[5], r[6], r[7], r[8], round(r[9], 1), r[10], r[11], r[12]]
                                             for r in rows], separators=(",", ":")))
        .replace("__PLAY__", json.dumps(plays)).replace("__MODEL__", ms["best_model"]).replace("__AUC__", f"{ms['oof_auc_all']:.2f}")
        .replace("__THR__", f"{ms['threshold']:.2f}").replace("__COST__", f"{a['contact_cost']:.0f}")
        .replace("__SUC__", f"{a['offer_success']*100:.0f}").replace("__MON__", str(a["months_saved"])))
(ROOT / "outputs" / "Churn_Dashboard.html").write_text(html, encoding="utf-8")
print("Dashboard written")
