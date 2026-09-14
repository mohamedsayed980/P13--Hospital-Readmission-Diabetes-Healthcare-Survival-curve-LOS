"""
Repo_13_Hospital_Readmission — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: Diabetes 130-US Hospitals · 101,763 patients · 1999–2008
"""
import os, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Hospital Readmission · M3",
                   page_icon="🏥", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"

_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "readmission_clean.csv"
)

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏥 EDA Dashboard")
    st.markdown("Hospital Readmission · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ readmission_clean.csv")
    st.caption("Loaded from data/ folder")

ACCENT = "#6a1b9a"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","grey":"#546e7a"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#f3e5f5;border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}}
</style>""", unsafe_allow_html=True)

def sec(t):     st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────
if not os.path.exists(_data_path):
    st.error(f"❌ File not found: {_data_path}")
    st.info("Run P13_clean_data.py in Jupyter → copy readmission_clean.csv to data/ folder")
    st.stop()

df = pd.read_csv(_data_path, sep=",", decimal=".", keep_default_na=False)
df.columns = df.columns.str.strip()
if df.empty:
    st.warning("⚠️ Dataset is empty."); st.stop()

S["df_work"] = df

TARGET  = "readmitted_30"
REG_T   = "time_in_hospital"

MED_COLS = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
            'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
            'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
            'examide','citoglipton','insulin','glyburide-metformin',
            'glipizide-metformin','glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone']

NUM_COLS = [c for c in ['age_mid','time_in_hospital','num_lab_procedures',
                         'num_procedures','num_medications','number_outpatient',
                         'number_emergency','number_inpatient','number_diagnoses',
                         'total_visits','total_med_changes','num_active_meds']
            if c in df.columns]

tabs = st.tabs([
    "1 · Data Overview",
    "2 · Readmission Analysis ★",
    "3 · Demographics ★",
    "4 · Hospital Utilization ★",
    "5 · Medication Analysis ★",
    "6 · Lab Results ★",
    "7 · Multicollinearity",
    "8 · Correlation",
    "9 · Business KPIs ★",
    "10 · Category Deep-Dive ★",
    "11 · Statistical Tests ★",
    "12 · Feature Engineering",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Patients",       f"{len(df):,}")
    c2.metric("Features",       f"{df.shape[1]}")
    c3.metric("Readmit <30d",   f"{df[TARGET].mean()*100:.1f}%")
    c4.metric("Mean Age",       f"{df['age_mid'].mean():.1f} yrs")
    c5.metric("Mean Stay",      f"{df[REG_T].mean():.1f} days")

    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({"Column":df.columns,
                                 "Dtype":df.dtypes.astype(str).values,
                                 "Nulls":df.isnull().sum().values})
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics — Numeric Columns")
    st.dataframe(df[NUM_COLS].describe().round(4), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column":["readmitted_30","readmitted_any","time_in_hospital","age_mid",
                  "num_lab_procedures","num_procedures","num_medications",
                  "number_outpatient","number_emergency","number_inpatient",
                  "number_diagnoses","high_utilization","long_stay",
                  "uses_insulin","is_senior","emergency_admission",
                  "total_visits","total_med_changes","num_active_meds"],
        "Type":  ["Target(CLF)","Derived","Target(REG)","Engineered",
                  "Numeric","Numeric","Numeric","Numeric","Numeric","Numeric",
                  "Numeric","Engineered","Engineered","Engineered","Engineered",
                  "Engineered","Engineered","Engineered","Engineered"],
        "Description":[
            "1 if readmitted within 30 days — PRIMARY CLF TARGET",
            "1 if readmitted at any time (30d or >30d)",
            "Days in hospital (1–14) — REGRESSION TARGET",
            "Age band midpoint (5–95 years)",
            "Number of lab tests performed",
            "Number of non-lab procedures",
            "Number of medications prescribed",
            "Number of outpatient visits in past year",
            "Number of emergency visits in past year",
            "Number of inpatient visits in past year",
            "Number of diagnoses recorded",
            "1 if total_visits > Q75 — 18.93% vs 9.72% readmit",
            "1 if time_in_hospital >= 7 days",
            "1 if insulin_enc >= 1 (any insulin use)",
            "1 if age_mid >= 65",
            "1 if admission_type_id == 1 (Emergency)",
            "outpatient + emergency + inpatient visits",
            "Count of medication Up/Down changes",
            "Count of medications on Steady or Up",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    warn("11.2% readmission rate → SEVERE IMBALANCE → class_weight='balanced' MANDATORY")

# ══════════════════════════════════════════════════════════════
# TAB 2 — READMISSION ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("🏥 Tab 2 — Readmission Analysis ★")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Readmit <30d",  f"{df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")
    c2.metric("Readmit Any",   f"{df['readmitted_any'].sum():,} ({df['readmitted_any'].mean()*100:.1f}%)")
    c3.metric("Not Readmitted",f"{(df['readmitted']=='NO').sum():,}")
    c4.metric("Mean Stay",     f"{df[REG_T].mean():.1f} days")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Readmission Category Distribution")
        read_cnt = df['readmitted'].value_counts()
        fig = px.bar(x=read_cnt.index, y=read_cnt.values,
                     color=read_cnt.index,
                     color_discrete_map={'NO':CLR['success'],'<30':CLR['danger'],'>30':CLR['warning']},
                     title="Readmission Categories",
                     text=read_cnt.values)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Days in Hospital Distribution")
        fig2, ax = plt.subplots(figsize=(7,4))
        ax.hist(df[REG_T], bins=14, color=CLR['purple'], edgecolor='white', alpha=0.85)
        ax.axvline(df[REG_T].mean(), color=CLR['danger'], lw=2.5, ls='--',
                   label=f"Mean={df[REG_T].mean():.1f}")
        ax.axvline(7, color=CLR['warning'], lw=2, ls=':', label='Long stay=7d')
        ax.set_xlabel("Days in Hospital"); ax.set_ylabel("Patient Count")
        ax.set_title("Length of Stay Distribution", fontweight='bold')
        ax.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Readmission Rate by Admission Type")
    adm_type = {1:'Emergency',2:'Urgent',3:'Elective',4:'Newborn',5:'Not Available',
                6:'NULL',7:'Trauma Center',8:'Not Mapped'}
    df['adm_label'] = df['admission_type_id'].map(adm_type).fillna('Other')
    adm_rate = df.groupby('adm_label')[TARGET].mean() * 100
    adm_rate = adm_rate.sort_values(ascending=False)
    fig3 = px.bar(x=adm_rate.index, y=adm_rate.values,
                  color=adm_rate.values,
                  color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                  title="Readmission Rate % by Admission Type",
                  text=adm_rate.values.round(1))
    fig3.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
    fig3.add_hline(y=df[TARGET].mean()*100, line_dash="dash",
                   line_color=CLR['primary'],
                   annotation_text=f"Avg {df[TARGET].mean()*100:.1f}%")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

    insight("Trauma Center admissions show highest readmission risk — most severe cases.")
    insight("Elective admissions have lowest readmission rate — planned, optimised care.")
    warn("11.2% overall rate — never evaluate with accuracy. Use F1, Recall, ROC-AUC.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DEMOGRAPHICS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("👥 Tab 3 — Demographics ★")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Readmission Rate by Age Group")
        age_rate = df.groupby('age')[TARGET].mean() * 100
        age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                     '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
        age_rate = age_rate.reindex([a for a in age_order if a in age_rate.index])
        fig = px.bar(x=age_rate.index, y=age_rate.values,
                     color=age_rate.values,
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Readmission Rate % by Age Group",
                     text=age_rate.values.round(1))
        fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig.add_hline(y=df[TARGET].mean()*100, line_dash="dash",
                      line_color=CLR['primary'])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Readmission Rate by Race")
        race_rate = df.groupby('race')[TARGET].agg(
            Total='count', Readmit='sum').reset_index()
        race_rate['Rate%'] = (race_rate['Readmit']/race_rate['Total']*100).round(2)
        race_rate = race_rate.sort_values('Rate%', ascending=False)
        fig2 = px.bar(race_rate, x='race', y='Rate%',
                      color='Rate%',
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Readmission Rate % by Race",
                      text='Rate%')
        fig2.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Gender Distribution")
        gender_rate = df.groupby('gender')[TARGET].agg(
            Total='count', Readmit='sum').reset_index()
        gender_rate['Rate%'] = (gender_rate['Readmit']/gender_rate['Total']*100).round(2)
        fig3 = px.bar(gender_rate, x='gender', y='Rate%',
                      color='gender',
                      color_discrete_map={'Female':CLR['purple'],'Male':CLR['primary']},
                      title="Readmission Rate % by Gender",
                      text='Rate%')
        fig3.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("📊 Senior vs Non-Senior")
        sr = df.groupby('is_senior')[TARGET].mean() * 100
        fig4, ax = plt.subplots(figsize=(6,3))
        bars = ax.bar(['Non-Senior (<65)','Senior (≥65)'],
                      sr.values, color=[CLR['primary'],CLR['purple']],
                      edgecolor='white')
        ax.axhline(df[TARGET].mean()*100, color=CLR['danger'],
                   lw=2, ls='--', label=f"Avg {df[TARGET].mean()*100:.1f}%")
        ax.set_ylabel("Readmission Rate %")
        ax.set_title("Readmission: Senior vs Non-Senior", fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, sr.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.1,
                    f"{val:.1f}%", ha='center', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    insight(f"Mean age: {df['age_mid'].mean():.1f} years — elderly population dominates.")
    insight("AfricanAmerican patients show higher readmission rates — social determinants of health.")
    warn("Race disparities in readmission are complex — confounded by socioeconomic factors.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — HOSPITAL UTILIZATION ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🏨 Tab 4 — Hospital Utilization ★")
    info("Prior utilization history is the strongest readmission predictor — past behaviour predicts future behaviour.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("High Utilization Readmit",  "18.93%")
    c2.metric("Normal Utilization Readmit","9.72%")
    c3.metric("Long Stay Readmit",         "13.23%")
    c4.metric("Short Stay Readmit",        "10.62%")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Readmission by Utilization Level")
        hu = df.groupby('high_utilization')[TARGET].mean() * 100
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(['Normal Utilization','High Utilization'],
                      hu.values,
                      color=[CLR['success'],CLR['danger']],
                      edgecolor='white')
        ax.axhline(df[TARGET].mean()*100, color=CLR['primary'],
                   lw=2, ls='--', label=f"Avg {df[TARGET].mean()*100:.1f}%")
        ax.set_ylabel("Readmission Rate %")
        ax.set_title("High Utilization vs Normal", fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, hu.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.2,
                    f"{val:.1f}%", ha='center', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Total Visits Distribution — Readmitted vs Not")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        df[df[TARGET]==0]['total_visits'].clip(0,10).hist(
            bins=11, ax=ax2, alpha=0.6, color=CLR['primary'],
            density=True, label='Not Readmitted')
        df[df[TARGET]==1]['total_visits'].clip(0,10).hist(
            bins=11, ax=ax2, alpha=0.7, color=CLR['danger'],
            density=True, label='Readmitted <30d')
        ax2.set_xlabel('Total Prior Visits')
        ax2.set_ylabel('Density')
        ax2.set_title('Prior Visits: Readmitted vs Not', fontweight='bold')
        ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Readmission Rate by Number of Prior Inpatient Visits")
    inp_rate = df.groupby('number_inpatient')[TARGET].mean() * 100
    inp_rate = inp_rate[inp_rate.index <= 10]
    fig3 = px.bar(x=inp_rate.index, y=inp_rate.values,
                  color=inp_rate.values,
                  color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                  title="Readmission Rate % by Prior Inpatient Visits",
                  text=inp_rate.values.round(1),
                  labels={'x':'Prior Inpatient Visits','y':'Readmission Rate %'})
    fig3.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
    fig3.add_hline(y=df[TARGET].mean()*100, line_dash="dash",
                   line_color=CLR['primary'],
                   annotation_text=f"Avg {df[TARGET].mean()*100:.1f}%")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

    insight("High utilization patients have 18.93% vs 9.72% — nearly 2× readmission rate.")
    insight("Each additional prior inpatient visit significantly increases readmission probability.")
    warn("Emergency prior visits are the most concerning signal — unplanned = unstable condition.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — MEDICATION ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("💊 Tab 5 — Medication Analysis ★")
    info("23 diabetes medications tracked. Insulin use and medication changes are key clinical signals.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Insulin Users",       f"{df['uses_insulin'].sum():,} ({df['uses_insulin'].mean()*100:.1f}%)")
    c2.metric("Insulin Readmit",     "12.14%")
    c3.metric("On Diabetes Med",     f"{(df['diabetesMed']=='Yes').sum():,}")
    c4.metric("Avg Active Meds",     f"{df['num_active_meds'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Insulin Use — Readmission Rate")
        ins_rate = df.groupby('insulin')[TARGET].mean() * 100
        fig = px.bar(x=ins_rate.index, y=ins_rate.values,
                     color=ins_rate.values,
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Readmission Rate % by Insulin Status",
                     text=ins_rate.values.round(1))
        fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig.add_hline(y=df[TARGET].mean()*100, line_dash="dash",
                      line_color=CLR['primary'])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Medication Change Impact")
        chg_rate = df.groupby('change')[TARGET].mean() * 100
        fig2, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(['No Change','Medication Changed'],
                      chg_rate.values,
                      color=[CLR['success'],CLR['warning']],
                      edgecolor='white')
        ax.axhline(df[TARGET].mean()*100, color=CLR['danger'],
                   lw=2, ls='--', label=f"Avg {df[TARGET].mean()*100:.1f}%")
        ax.set_ylabel("Readmission Rate %")
        ax.set_title("Medication Change vs No Change", fontweight='bold')
        ax.legend()
        for bar, val in zip(bars, chg_rate.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.1,
                    f"{val:.1f}%", ha='center', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Top 10 Medications by Usage Rate")
    med_usage = {m: (df[m] != 'No').mean()*100 for m in MED_COLS if m in df.columns}
    med_df = pd.DataFrame(list(med_usage.items()),
                          columns=['Medication','Usage%'])\
               .sort_values('Usage%', ascending=False).head(10)
    fig3 = px.bar(med_df, x='Medication', y='Usage%',
                  color='Usage%',
                  color_continuous_scale=["#e3f2fd","#1565c0"],
                  title="Top 10 Medications by Usage Rate %",
                  text=med_df['Usage%'].round(1))
    fig3.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

    insight("Insulin is by far the most used medication — 53.5% of patients.")
    insight("Patients with medication changes have different readmission rates — dose adjustments signal instability.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — LAB RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🧪 Tab 6 — Lab Results ★")
    info("Glucose serum and A1C are key diabetes control indicators. Most patients had NO test taken.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Glucose Serum Result — Readmission Rate")
        glu_rate = df.groupby('max_glu_serum')[TARGET].agg(
            Total='count', Readmit='sum').reset_index()
        glu_rate['Rate%'] = (glu_rate['Readmit']/glu_rate['Total']*100).round(2)
        glu_rate = glu_rate.sort_values('Rate%', ascending=False)
        fig = px.bar(glu_rate, x='max_glu_serum', y='Rate%',
                     color='Rate%',
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Readmission Rate % by Glucose Serum",
                     text='Rate%')
        fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig.update_layout(height=370)
        st.plotly_chart(fig, use_container_width=True)

        glu_counts = df['max_glu_serum'].value_counts()
        st.info(f"Note: 'None' = {glu_counts.get('None',0):,} patients had NO glucose test ({glu_counts.get('None',0)/len(df)*100:.0f}%)")

    with col2:
        sec("📊 A1C Result — Readmission Rate")
        a1c_rate = df.groupby('A1Cresult')[TARGET].agg(
            Total='count', Readmit='sum').reset_index()
        a1c_rate['Rate%'] = (a1c_rate['Readmit']/a1c_rate['Total']*100).round(2)
        a1c_rate = a1c_rate.sort_values('Rate%', ascending=False)
        fig2 = px.bar(a1c_rate, x='A1Cresult', y='Rate%',
                      color='Rate%',
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Readmission Rate % by A1C Result",
                      text='Rate%')
        fig2.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig2.update_layout(height=370)
        st.plotly_chart(fig2, use_container_width=True)

        a1c_counts = df['A1Cresult'].value_counts()
        st.info(f"Note: 'None' = {a1c_counts.get('None',0):,} patients had NO A1C test ({a1c_counts.get('None',0)/len(df)*100:.0f}%)")

    insight("A1C >8 indicates poor long-term glucose control — higher readmission risk expected.")
    warn("94.7% had no glucose test and 83.3% had no A1C test — test ordering bias may affect results.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("🔁 Tab 7 — Multicollinearity / VIF")

    vif_cols = [c for c in ['age_mid','time_in_hospital','num_lab_procedures',
                              'num_procedures','num_medications','number_outpatient',
                              'number_emergency','number_inpatient','number_diagnoses',
                              'total_visits','total_med_changes','num_active_meds',
                              'high_utilization','long_stay','uses_insulin',
                              'is_senior','emergency_admission']
                if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values,i),2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1: st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,6))
            colors_vif = [CLR['danger'] if v>10 else CLR['warning'] if v>5
                          else CLR['success'] for v in vif_df['VIF']]
            ax.barh(vif_df['Feature'], vif_df['VIF'], color=colors_vif)
            ax.axvline(10, color=CLR['danger'],  lw=2, ls='--', label='VIF=10')
            ax.axvline(5,  color=CLR['warning'], lw=1.5, ls=':', label='VIF=5')
            ax.set_xlabel("VIF"); ax.set_title("Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("total_visits is derived from outpatient+emergency+inpatient — expected high VIF.")
    insight("Tree models (RF, GB) handle multicollinearity — preferred for this dataset.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    num_corr = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['readmitted_any','admission_type_id',
                             'discharge_disposition_id','admission_source_id']]
    corr = df[num_corr].corr()

    fig, ax = plt.subplots(figsize=(14,10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":6})
    ax.set_title("Correlation Matrix — Hospital Readmission Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec(f"🎯 Top Correlations with {TARGET}")
    tgt = corr[TARGET].drop(TARGET).sort_values(key=abs,ascending=False).head(12)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors_bar = [CLR['danger'] if v>0 else CLR['success'] for v in tgt.values]
    ax2.barh(tgt.index, tgt.values, color=colors_bar)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel(f"Pearson r with {TARGET}")
    ax2.set_title("Feature Correlations with 30-Day Readmission", fontsize=12, fontweight='bold')
    for i,(idx,val) in enumerate(tgt.items()):
        ax2.text(val+0.002 if val>=0 else val-0.002, i,
                 f"{val:.3f}", va='center',
                 ha='left' if val>=0 else 'right', fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("number_inpatient and high_utilization show strongest correlation with readmission.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💼 Tab 9 — Business KPIs ★")

    COST_PER_READMISSION  = 15000
    COST_INTERVENTION     = 500
    READMISSION_PREVENTED = 0.30

    total_readmit    = df[TARGET].sum()
    total_cost       = total_readmit * COST_PER_READMISSION
    high_util_readmit= df[(df['high_utilization']==1) & (df[TARGET]==1)].shape[0]
    intervention_cost= total_readmit * COST_INTERVENTION
    prevented        = int(total_readmit * READMISSION_PREVENTED)
    savings          = prevented * COST_PER_READMISSION - intervention_cost
    roi              = savings / intervention_cost * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Readmissions",    f"{total_readmit:,}")
    c2.metric("Annual Cost Est.",      f"${total_cost/1e6:.1f}M")
    c3.metric("Potential Savings",     f"${savings/1e6:.1f}M")
    c4.metric("Intervention ROI",      f"{roi:,.0f}%")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Cost Breakdown")
        cost_df = pd.DataFrame({
            "Category":["Total Readmission Cost","Intervention Cost","Net Savings"],
            "Amount ($M)":[total_cost/1e6, intervention_cost/1e6, savings/1e6]
        })
        fig = px.bar(cost_df, x='Category', y='Amount ($M)',
                     color='Category',
                     color_discrete_map={"Total Readmission Cost":CLR['danger'],
                                         "Intervention Cost":CLR['warning'],
                                         "Net Savings":CLR['success']},
                     title="Business Case: Prevention vs Status Quo",
                     text=cost_df['Amount ($M)'].apply(lambda x: f"${x:.1f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 High-Risk Segment Analysis")
        segments = {
            "High Utilization": df[df['high_utilization']==1][TARGET].sum(),
            "Long Stay":        df[df['long_stay']==1][TARGET].sum(),
            "Senior + Insulin": df[(df['is_senior']==1)&(df['uses_insulin']==1)][TARGET].sum(),
            "Emergency Admit":  df[df['emergency_admission']==1][TARGET].sum(),
        }
        seg_df = pd.DataFrame(list(segments.items()),
                              columns=['Segment','Readmissions'])\
                   .sort_values('Readmissions', ascending=True)
        fig2, ax = plt.subplots(figsize=(7,4))
        ax.barh(seg_df['Segment'], seg_df['Readmissions'],
                color=CLR['purple'], edgecolor='white')
        ax.set_xlabel('Readmission Count')
        ax.set_title('Readmissions by High-Risk Segment', fontweight='bold')
        for i, v in enumerate(seg_df['Readmissions']):
            ax.text(v+50, i, f"{v:,}", va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight(f"Targeting high-utilization patients alone could prevent {high_util_readmit:,} readmissions.")
    insight(f"Even at 30% prevention rate, intervention saves ${savings/1e6:.1f}M annually at {roi:,.0f}% ROI.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — CATEGORY DEEP-DIVE ★
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🔎 Tab 10 — Category Deep-Dive ★")

    sec("📊 Age Group × Insulin Use — Readmission Rate")
    df['age_group_5'] = pd.cut(df['age_mid'],
        bins=[0,40,55,65,75,100],
        labels=['<40','40-55','55-65','65-75','75+'])
    heat = df.groupby(['age_group_5','insulin'],
                       observed=True)[TARGET].mean().unstack() * 100
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(heat.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5, annot_kws={"size":10})
        ax.set_title("Readmission Rate % — Age Group × Insulin Status",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Race × is_senior — Readmission Rate")
        heat2 = df.groupby(['race','is_senior'])[TARGET].mean().unstack() * 100
        heat2.columns = ['Non-Senior','Senior']
        fig2, ax2 = plt.subplots(figsize=(7,5))
        sns.heatmap(heat2.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax2, linewidths=0.5, annot_kws={"size":10})
        ax2.set_title("Readmission % — Race × Senior Status", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    with col2:
        sec("📊 High Utilization × Long Stay")
        heat3 = df.groupby(['high_utilization','long_stay'])[TARGET].mean().unstack() * 100
        heat3.index = ['Normal Util.','High Util.']
        heat3.columns = ['Short Stay','Long Stay']
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.heatmap(heat3.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax3, linewidths=0.5, annot_kws={"size":14})
        ax3.set_title("Readmission % — Utilization × Stay Length", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("High utilization + long stay = worst combination — most intensive case management needed.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — STATISTICAL TESTS ★
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("🧪 Tab 11 — Statistical Tests ★")

    def run_test(gA, gB, lA, lB):
        t_stat, p_val = stats.ttest_ind(gA.dropna(), gB.dropna(), equal_var=False)
        pooled   = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d = (gA.mean() - gB.mean()) / (pooled + 1e-10)
        res = pd.DataFrame({
            "Metric":["Test","Group A","Group B","A Mean","B Mean",
                      "t-stat","p-value","Significant","Cohen's d","Effect","Decision"],
            "Result":["Welch T-Test",
                      f"{lA} (n={len(gA):,})",f"{lB} (n={len(gB):,})",
                      f"{gA.mean():.4f}",f"{gB.mean():.4f}",
                      f"{t_stat:.4f}",f"{p_val:.6f}",
                      "✅ YES" if p_val<0.05 else "❌ NO",
                      f"{cohens_d:.4f}",
                      "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                      "✅ REJECT H₀" if p_val<0.05 else "❌ FAIL"]
        })
        return res, p_val, cohens_d

    sec("T1 — Do high-utilization patients have significantly more readmissions?")
    r1,p1,d1 = run_test(
        df[df['high_utilization']==1][TARGET].astype(float),
        df[df['high_utilization']==0][TARGET].astype(float),
        "High Utilization","Normal Utilization")
    st.dataframe(r1, use_container_width=True)
    if p1 < 0.05:
        insight(f"T1: High utilization significantly predicts readmission (d={d1:.3f}) — target this group first.")

    st.markdown("---")
    sec("T2 — Do long-stay patients have more lab procedures than short-stay?")
    r2,p2,d2 = run_test(
        df[df['long_stay']==1]['num_lab_procedures'],
        df[df['long_stay']==0]['num_lab_procedures'],
        "Long Stay (≥7d)","Short Stay (<7d)")
    st.dataframe(r2, use_container_width=True)
    if p2 < 0.05:
        insight(f"T2: Long-stay patients have significantly more lab procedures (d={d2:.3f}).")

    st.markdown("---")
    sec("T3 — Do seniors spend significantly longer in hospital?")
    r3,p3,d3 = run_test(
        df[df['is_senior']==1][REG_T],
        df[df['is_senior']==0][REG_T],
        "Senior (≥65)","Non-Senior (<65)")
    st.dataframe(r3, use_container_width=True)
    if p3 < 0.05:
        insight(f"T3: Senior patients have significantly longer hospital stays (d={d3:.3f}).")

    st.markdown("---")
    sec("T4 — Does medication change significantly affect time in hospital?")
    r4,p4,d4 = run_test(
        df[df['change']=='Ch'][REG_T],
        df[df['change']=='No'][REG_T],
        "Medication Changed","No Change")
    st.dataframe(r4, use_container_width=True)
    if p4 < 0.05:
        insight(f"T4: Medication changes associated with significantly different stay length (d={d4:.3f}).")

# ══════════════════════════════════════════════════════════════
# TAB 12 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("⚙️ Tab 12 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":        ["age_mid","total_visits","total_med_changes","num_active_meds",
                           "high_utilization","long_stay","uses_insulin",
                           "is_senior","emergency_admission","readmitted_30"],
        "Formula":        ["Age band midpoint (e.g. [70-80)→75)",
                           "outpatient + emergency + inpatient",
                           "Count of Up+Down medication changes",
                           "Count of Steady+Up medications",
                           "total_visits > Q75",
                           "time_in_hospital ≥ 7",
                           "insulin_enc ≥ 1",
                           "age_mid ≥ 65",
                           "admission_type_id == 1",
                           "(readmitted == '<30').astype(int)"],
        "Clinical Reason":["Ordinal age for ML correlation",
                           "Overall healthcare utilization history",
                           "Medication instability indicator",
                           "Medication burden — more = sicker",
                           "Top 25% utilizers — 2× readmit risk",
                           "Serious complications — higher discharge risk",
                           "Advanced diabetes — harder to control",
                           "Multiple comorbidities — slower recovery",
                           "Unplanned deterioration — more unstable",
                           "Primary classification target"],
        "Readmit Impact": ["Older → higher risk",
                           "18.93% vs 9.72% (+9.2 pp) ★★★",
                           "More changes → more instability",
                           "More meds → more comorbidities",
                           "2× readmission rate",
                           "13.23% vs 10.62% (+2.6 pp)",
                           "12.14% vs 10.04% (+2.1 pp)",
                           "11.61% vs 10.23% (+1.4 pp)",
                           "11.52% vs 10.75% (+0.8 pp)",
                           "11.2% positive rate"],
    })
    st.dataframe(fe, use_container_width=True)

    insight("high_utilization is the single strongest engineered feature — 2× readmission rate.")
    warn("age encoded as midpoint — ordinal relationship preserved for ML correlation.")
    warn("'None' in max_glu_serum and A1Cresult = test not ordered, not missing data.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### 🏥 Hospital Readmission — Final Report")
    st.markdown(f"**101,763 patients · 130 US Hospitals · 1999–2008 · M3**")
    st.markdown("---")

    sec("1️⃣ Readmission Profile")
    insight(f"11.2% of patients readmitted within 30 days = {df[TARGET].sum():,} patients.")
    insight("46.1% readmitted at any time — long-term diabetes management is challenging.")
    warn("30-day readmission is the clinical and CMS quality metric — this is the key target.")

    sec("2️⃣ Top Risk Factors")
    insight("High utilization: 18.93% vs 9.72% — nearly 2× readmission rate. #1 predictor.")
    insight("Long stay ≥7 days: 13.23% vs 10.62% — serious inpatient complications.")
    insight("Insulin use: 12.14% vs 10.04% — advanced disease, harder glucose control.")
    insight("Senior patients (≥65): 11.61% vs 10.23% — multiple comorbidities slow recovery.")

    sec("3️⃣ Recommendations")
    recs = [
        ("📞 High Utilization Protocol",
         "Enroll all high-utilization patients (top 25% visits) in post-discharge call program within 48 hours."),
        ("💊 Medication Reconciliation",
         "Assign pharmacist review for all patients with ≥3 medication changes during admission."),
        ("👴 Senior Care Pathway",
         "Activate enhanced discharge planning for all patients ≥65 — social work, home care assessment."),
        ("🩺 Insulin Education",
         "Mandatory diabetes educator consultation for all insulin-dependent patients before discharge."),
        ("🚨 Emergency Admission Tracking",
         "Flag all emergency admissions for 72-hour follow-up call — highest instability risk."),
        ("📊 Predictive Scoring",
         "Deploy ML readmission risk score at admission — target interventions to top 20% risk patients."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""HOSPITAL READMISSION — FINAL REPORT
M3 · {len(df):,} Patients · 130 US Hospitals · 1999-2008

TARGET: 30-day readmission rate = {df[TARGET].mean()*100:.1f}%
        ({df[TARGET].sum():,} patients readmitted within 30 days)

TOP RISK FACTORS:
1. High utilization : 18.93% vs 9.72% (nearly 2×) ← #1 predictor
2. Long stay ≥7 days: 13.23% vs 10.62% (+2.6 pp)
3. Insulin use      : 12.14% vs 10.04% (+2.1 pp)
4. Senior (age ≥65) : 11.61% vs 10.23% (+1.4 pp)
5. Emergency admit  : 11.52% vs 10.75% (+0.8 pp)

RECOMMENDATIONS:
1. 48-hour post-discharge call for high-utilization patients
2. Pharmacist review for patients with ≥3 medication changes
3. Enhanced discharge planning for all patients ≥65
4. Diabetes educator for all insulin-dependent patients
5. 72-hour follow-up for all emergency admissions
6. Deploy ML risk score for proactive intervention targeting
"""

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="HospitalReadmission_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        # High-risk patient segment summary
        seg_summary = df.groupby(['is_senior','uses_insulin','high_utilization'])\
                        .agg(Patients=('readmitted_30','count'),
                             Readmissions=('readmitted_30','sum'))\
                        .reset_index()
        seg_summary['Rate%'] = (seg_summary['Readmissions']/
                                 seg_summary['Patients']*100).round(2)
        seg_summary['is_senior']       = seg_summary['is_senior'].map({0:'No',1:'Yes'})
        seg_summary['uses_insulin']    = seg_summary['uses_insulin'].map({0:'No',1:'Yes'})
        seg_summary['high_utilization']= seg_summary['high_utilization'].map({0:'No',1:'Yes'})
        seg_summary = seg_summary.sort_values('Rate%', ascending=False)
        st.download_button("📥 Risk Segment Analysis (.csv)",
                           seg_summary.to_csv(index=False),
                           file_name="ReadmissionRisk_Segments_M3.csv",
                           mime="text/csv", use_container_width=True)
    with col3:
        # Medication impact summary
        med_impact = pd.DataFrame([{
            'Medication': m,
            'Users': (df[m]!='No').sum(),
            'Usage%': round((df[m]!='No').mean()*100,2),
            'Readmit_Rate%': round(df[df[m]!='No'][TARGET].mean()*100,2)
        } for m in MED_COLS if m in df.columns])\
        .sort_values('Users', ascending=False)
        st.download_button("📥 Medication Impact (.csv)",
                           med_impact.to_csv(index=False),
                           file_name="MedicationImpact_M3.csv",
                           mime="text/csv", use_container_width=True)

    st.markdown("---")
    sec("📊 Preview — Risk Segment Analysis")
    st.dataframe(seg_summary.head(10).style
                 .background_gradient(subset=['Rate%'], cmap='RdYlGn_r')
                 .format({'Rate%':'{:.2f}%'}),
                 use_container_width=True)
