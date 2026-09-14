"""
Repo_13_Hospital_Readmission — Home.py
Author : Mohamed · M3
"""
import pathlib
import streamlit as st

st.set_page_config(page_title="Hospital Readmission · M3",
                   page_icon="🏥", layout="wide")
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏥 Hospital Readmission")
    st.markdown("M3 · ML Engine · P13")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

ACCENT = "#6a1b9a"

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
.hero{{background:linear-gradient(135deg,#1a237e,{ACCENT});
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}}
.hero h1{{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}}
.hero p{{color:#e1bee7 !important;font-size:1.08rem;margin:0;}}
.card{{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid {ACCENT};}}
.card h3{{color:{ACCENT} !important;margin:0 0 8px 0;font-size:1.05rem;}}
.card p{{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}}
.stat-card{{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid {ACCENT};}}
.stat-num{{font-size:1.9rem;font-weight:800;color:{ACCENT} !important;}}
.stat-lbl{{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏥 Hospital Readmission Prediction</h1>
  <p>101,763 diabetic patients · 130 US Hospitals · 1999–2008 · M3 Portfolio · Project 13</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5,c6],[
    ("101,763","Patients"), ("130","Hospitals"),
    ("10 Yrs","1999–2008"), ("11.2%","Readmit <30d"),
    ("66.0","Mean Age"), ("4.4 Days","Avg Stay")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="card"><h3>🎯 Objective</h3>
    <p>Predict which diabetic patients will be readmitted to hospital
    within 30 days of discharge. Early identification enables
    targeted interventions — follow-up calls, medication review,
    and care coordination — reducing costly readmissions.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="card"><h3>📊 Dataset</h3>
    <p>Diabetes 130-US Hospitals 1999–2008 · UCI Repository.
    101,763 inpatient encounters · 82 features after engineering.
    Covers: demographics, diagnoses, medications, lab results,
    hospital utilization history.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="card"><h3>🔑 Key Signals</h3>
    <p>High utilization patients: 18.93% vs 9.72% readmit rate (2×).
    Long stay ≥7 days: 13.23% vs 10.62%.
    Insulin use: 12.14% vs 10.04%.
    Emergency admission: 11.52% vs 10.75%.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","Readmission Analysis ★","30-day vs >30d vs No readmission"),
        ("3","Demographics ★","Age, gender, race breakdown"),
        ("4","Hospital Utilization ★","Visits, stay length, procedures"),
        ("5","Medication Analysis ★","23 medications + insulin deep-dive"),
        ("6","Lab Results ★","Glucose serum + A1C result analysis"),
        ("7","Multicollinearity","VIF analysis"),
        ("8","Correlation","Heatmap + top readmission predictors"),
        ("9","Business KPIs ★","Cost of readmissions · intervention ROI"),
        ("10","Category Deep-Dive ★","Race × Age × Admission type heatmaps"),
        ("11","Statistical Tests ★","T1-T4: utilization · age · stay length"),
        ("12","Feature Engineering","Engineered features + clinical logic"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict days in hospital"),
        ("3","Classification Results","F1, Recall, ROC-AUC · predict readmission"),
        ("4","Feature Importance","Top readmission risk predictors"),
        ("5","Predict","Interactive patient risk scorer"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("**Readmission rate: 11.2%** — SEVERE imbalance.\n\n"
               "class_weight='balanced' MANDATORY.\n\n"
               "Evaluate: F1 + Recall + ROC-AUC.\n\n"
               "Missing a high-risk patient = costly readmission.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 13 · Healthcare Analytics</p>",
            unsafe_allow_html=True)
