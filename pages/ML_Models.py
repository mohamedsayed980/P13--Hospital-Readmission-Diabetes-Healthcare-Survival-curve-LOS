"""
Repo_13_Hospital_Readmission — ML_Models.py  (5 Tabs)
Author : Mohamed · M3
Regression     → time_in_hospital (1–14 days)
Classification → readmitted_30 (1 if readmitted <30 days) — class_weight='balanced'
"""
import os, pathlib, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import psutil

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model      import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree              import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestRegressor, GradientBoostingRegressor,
                                       RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm               import LinearSVC
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.neighbors         import KNeighborsClassifier

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="ML Models · Hospital Readmission · M3",
                   page_icon="🤖", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"

_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "readmission_clean.csv"
)

ACCENT = "#6a1b9a"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🤖 ML Models")
    st.markdown("Hospital Readmission · 5 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ readmission_clean.csv")
    st.divider()
    st.markdown("### ⚙️ Options")
    test_size    = st.slider("Test Split %", 10, 40, 20, 5) / 100
    use_parallel = st.checkbox("Parallel (n_jobs=-1)", value=True)
    n_jobs       = -1 if use_parallel else 1
    st.warning("⚠️ Readmission rate: 11.2%\n\n"
               "class_weight='balanced'\nMANDATORY on ALL classifiers\n\n"
               "Evaluate: F1 + Recall + AUC\n"
               "Missing high-risk patient = costly readmission!")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","grey":"#546e7a"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{{background:{ACCENT} !important;color:#ffffff !important;border:none !important;border-radius:6px !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#f3e5f5;border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;}}
.info-box{{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#0d2a4a !important;margin:0;font-size:0.93rem;}}
</style>""", unsafe_allow_html=True)

def sec(t):     st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

def get_cpu_info(use_parallel, n_jobs):
    return {"total": os.cpu_count(), "used": n_jobs if use_parallel else 1,
            "percent": psutil.cpu_percent(interval=0.3)}

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

REG_TARGET = "time_in_hospital"
CLF_TARGET = "readmitted_30"
readmit_rate = df[CLF_TARGET].mean() * 100

# ── FEATURES ─────────────────────────────────────────────────
FEATURE_COLS = [c for c in [
    "age_mid","num_lab_procedures","num_procedures","num_medications",
    "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
    "total_visits","total_med_changes","num_active_meds",
    "high_utilization","long_stay","uses_insulin","is_senior","emergency_admission",
    "race_enc","gender_enc","max_glu_serum_enc","A1Cresult_enc",
    "change_enc","diabetesMed_enc",
    "insulin_enc","metformin_enc","glipizide_enc","glyburide_enc",
    "pioglitazone_enc","rosiglitazone_enc","glimepiride_enc",
    "admission_type_id","discharge_disposition_id","admission_source_id",
] if c in df.columns]

df_ml = df[FEATURE_COLS + [REG_TARGET, CLF_TARGET]].copy()
# Convert all to numeric safely
for col in FEATURE_COLS + [REG_TARGET, CLF_TARGET]:
    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
df_ml = df_ml.dropna()

X     = df_ml[FEATURE_COLS]
y_reg = df_ml[REG_TARGET]
y_clf = df_ml[CLF_TARGET].astype(int)

X_tr_r, X_te_r, yr_tr, yr_te = train_test_split(
    X, y_reg, test_size=test_size, random_state=42)
X_tr_c, X_te_c, yc_tr, yc_te = train_test_split(
    X, y_clf, test_size=test_size, random_state=42, stratify=y_clf)

scaler   = StandardScaler()
Xtr_r_sc = scaler.fit_transform(X_tr_r)
Xte_r_sc = scaler.transform(X_te_r)
Xtr_c_sc = scaler.fit_transform(X_tr_c)
Xte_c_sc = scaler.transform(X_te_c)

CW = "balanced"  # 11.2% readmission rate

REG_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Lasso":             Lasso(alpha=0.01, max_iter=5000),
    "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=CW,
                                               n_jobs=n_jobs, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, class_weight=CW,
                                                   random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight=CW,
                                                   n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)":        CalibratedClassifierCV(
                               LinearSVC(class_weight=CW, max_iter=3000, random_state=42)),
    "KNN":                 KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs),
}

tabs = st.tabs(["1 · Model Training",
                "2 · Regression Results",
                "3 · Classification Results",
                "4 · Feature Importance",
                "5 · Predict — Patient Risk"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🚀 Tab 1 — Model Training")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Patients",      f"{len(df_ml):,}")
    c2.metric("Features",      f"{len(FEATURE_COLS)}")
    c3.metric("Train Size",    f"{len(X_tr_r):,}")
    c4.metric("Test Size",     f"{len(X_te_r):,}")
    c5.metric("Readmit Rate",  f"{readmit_rate:.1f}%")

    cpu = get_cpu_info(use_parallel, n_jobs)
    st.info(f"🖥 CPU: {cpu['total']} cores · Using: {cpu['used']} · Load: {cpu['percent']}%")

    col1, col2 = st.columns(2)
    with col1:
        sec("📈 Regression Target")
        st.markdown(f"**`{REG_TARGET}`** — days in hospital (1–14)")
        st.markdown(f"Mean={y_reg.mean():.2f} · Median={y_reg.median():.0f} · Range=1–14")
    with col2:
        sec("🎯 Classification Target")
        st.markdown(f"**`{CLF_TARGET}`** — 1 if readmitted within 30 days")
        st.markdown(f"Readmit rate={readmit_rate:.1f}% → class_weight='balanced'")

    if "reg_results" not in S: S["reg_results"] = []
    if "reg_models"  not in S: S["reg_models"]  = {}
    if "clf_results" not in S: S["clf_results"] = []
    if "clf_models"  not in S: S["clf_models"]  = {}
    S["X_te_r"] = X_te_r; S["Xte_r_sc"] = Xte_r_sc
    S["X_te_c"] = X_te_c; S["Xte_c_sc"] = Xte_c_sc
    S["yr_te"]  = yr_te;  S["yc_te"]    = yc_te
    S["scaler"] = scaler; S["X_cols"]   = FEATURE_COLS

    def _done_r(n): return any(r["Model"]==n for r in S["reg_results"])
    def _done_c(n): return any(r["Model"]==n for r in S["clf_results"])

    def _train_reg(name, model):
        use_sc = name in ["Linear Regression","Ridge","Lasso"]
        Xtr = Xtr_r_sc if use_sc else X_tr_r
        Xte = Xte_r_sc if use_sc else X_te_r
        t0  = time.time(); model.fit(Xtr, yr_tr); preds = model.predict(Xte)
        row = {"Model":name,
               "R²":   round(r2_score(yr_te, preds),4),
               "MAE":  round(mean_absolute_error(yr_te, preds),4),
               "RMSE": round(np.sqrt(mean_squared_error(yr_te, preds)),4),
               "Time(s)": round(time.time()-t0,2)}
        S["reg_results"] = [r for r in S["reg_results"] if r["Model"]!=name] + [row]
        S["reg_models"][name] = model
        return row

    def _train_clf(name, model):
        use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
        Xtr = Xtr_c_sc if use_sc else X_tr_c
        Xte = Xte_c_sc if use_sc else X_te_c
        t0  = time.time(); model.fit(Xtr, yc_tr); preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        row = {"Model":name,
               "F1":        round(f1_score(yc_te, preds, zero_division=0),4),
               "Recall":    round(recall_score(yc_te, preds, zero_division=0),4),
               "Precision": round(precision_score(yc_te, preds, zero_division=0),4),
               "Accuracy":  round(accuracy_score(yc_te, preds),4),
               "ROC-AUC":   round(roc_auc_score(yc_te,proba),4) if proba is not None else 0.0,
               "Time(s)":   round(time.time()-t0,2)}
        S["clf_results"] = [r for r in S["clf_results"] if r["Model"]!=name] + [row]
        S["clf_models"][name] = model
        return row

    st.markdown("---")
    sec("📈 Regression Models — Train Individually")
    rc = st.columns(3)
    for i,(name,model) in enumerate(REG_MODELS.items()):
        with rc[i%3]:
            label = f"✅ {name}" if _done_r(name) else f"▶ Train {name}"
            if st.button(label, key=f"reg_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_reg(name, model)
                st.success(f"R²={row['R²']:.4f} · MAE={row['MAE']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if _done_r(name):
                r = next(r for r in S["reg_results"] if r["Model"]==name)
                st.caption(f"R²={r['R²']:.4f} · MAE={r['MAE']:.4f}")

    if S["reg_results"]:
        st.dataframe(pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["R²"],cmap="RdYlGn")
                       .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)

    st.markdown("---")
    sec("🎯 Classification Models — Train Individually")
    warn("11.2% readmission rate → class_weight='balanced' · evaluate F1 + Recall + AUC only")
    cc = st.columns(3)
    for i,(name,model) in enumerate(CLF_MODELS.items()):
        with cc[i%3]:
            label = f"✅ {name}" if _done_c(name) else f"▶ Train {name}"
            if st.button(label, key=f"clf_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_clf(name, model)
                st.success(f"F1={row['F1']:.4f} · Recall={row['Recall']:.4f} · AUC={row['ROC-AUC']:.4f}")
                st.rerun()
            if _done_c(name):
                r = next(r for r in S["clf_results"] if r["Model"]==name)
                st.caption(f"F1={r['F1']:.4f} · Recall={r['Recall']:.4f}")

    if S["clf_results"]:
        st.dataframe(pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["F1","Recall","ROC-AUC"],cmap="RdYlGn")
                       .format({c:"{:.4f}" for c in ["F1","Recall","Precision","Accuracy","ROC-AUC"]}),
                     use_container_width=True)

    n_done = len(S["reg_results"]) + len(S["clf_results"])
    st.info(f"📊 {n_done}/12 models trained." if n_done<12
            else "✅ All 12 models trained! Navigate to Results tabs →")

# ══════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Regression Results")
    info("Predicting: **days in hospital** (1–14). Moderate R² expected — stay length depends on many clinical factors.")

    if not S.get("reg_results"):
        warn("Train at least one Regression model in Tab 1.")
    else:
        reg_df   = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        best_reg = reg_df.iloc[0]["Model"]

        st.dataframe(reg_df.style
                       .background_gradient(subset=["R²"],cmap="RdYlGn")
                       .background_gradient(subset=["MAE","RMSE"],cmap="RdYlGn_r")
                       .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_reg}` — R²={reg_df.iloc[0]['R²']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(reg_df, x="Model", y="R²",
                         color="R²", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="R² — All Regression Models",
                         text=reg_df["R²"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="MAE",  x=reg_df["Model"],
                                  y=reg_df["MAE"],  marker_color=CLR["warning"]))
            fig2.add_trace(go.Bar(name="RMSE", x=reg_df["Model"],
                                  y=reg_df["RMSE"], marker_color=CLR["danger"]))
            fig2.update_layout(barmode="group",height=370,
                                title="MAE vs RMSE",xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec(f"📈 Actual vs Predicted — {best_reg}")
        bm   = S["reg_models"][best_reg]
        Xte  = S["Xte_r_sc"] if best_reg in ["Linear Regression","Ridge","Lasso"] else S["X_te_r"]
        pred = bm.predict(Xte)
        col3, col4 = st.columns(2)
        with col3:
            fig3, ax = plt.subplots(figsize=(7,5))
            ax.scatter(yr_te, pred, alpha=0.2, s=5, color=CLR["purple"])
            lims = [1, 14]
            ax.plot(lims,lims,"r--",lw=2,label="Perfect fit")
            ax.set_xlabel("Actual Days"); ax.set_ylabel("Predicted Days")
            ax.set_title(f"Actual vs Predicted — {best_reg}"); ax.legend()
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        with col4:
            resid = yr_te.values - pred
            fig4, ax2 = plt.subplots(figsize=(7,5))
            ax2.scatter(pred, resid, alpha=0.2, s=5, color=CLR["teal"])
            ax2.axhline(0, color=CLR["danger"], lw=2, ls="--")
            ax2.set_xlabel("Predicted Days"); ax2.set_ylabel("Residual")
            ax2.set_title("Residual Plot")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        info("Lower R² expected for stay length — many confounding clinical factors beyond available features.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🎯 Tab 3 — Classification Results")
    warn("11.2% readmission — NEVER use accuracy. A model predicting all 'Not Readmitted' gets 88.8% accuracy but catches ZERO readmissions.")

    if not S.get("clf_results"):
        warn("Train at least one Classification model in Tab 1.")
    else:
        clf_df   = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        best_clf = clf_df.iloc[0]["Model"]
        yc_te_s  = S["yc_te"]

        st.dataframe(clf_df.style
                       .background_gradient(subset=["F1","Recall","ROC-AUC"],cmap="RdYlGn")
                       .format({c:"{:.4f}" for c in ["F1","Recall","Precision","Accuracy","ROC-AUC"]}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_clf}` — F1={clf_df.iloc[0]['F1']:.4f} · AUC={clf_df.iloc[0]['ROC-AUC']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(clf_df, x="Model", y="F1",
                         color="F1", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="F1 Score — All Classifiers",
                         text=clf_df["F1"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(clf_df, x="Model", y="Recall",
                          color="Recall",
                          color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="Recall — Catching Readmissions",
                          text=clf_df["Recall"].apply(lambda x: f"{x:.4f}"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        bm      = S["clf_models"][best_clf]
        use_sc  = best_clf in ["Logistic Regression","SVM (Linear)","KNN"]
        Xte_c   = S["Xte_c_sc"] if use_sc else S["X_te_c"]
        preds_c = bm.predict(Xte_c)
        cm      = confusion_matrix(yc_te_s, preds_c)

        with col3:
            sec(f"🔢 Confusion Matrix — {best_clf}")
            fig3, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Not Readmitted","Readmitted <30d"],
                        yticklabels=["Not Readmitted","Readmitted <30d"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_clf}")
            plt.tight_layout(); st.pyplot(fig3); plt.close()
            tn,fp,fn,tp = cm.ravel()
            st.markdown(f"**TP={tp:,}** patients correctly flagged · **FN={fn:,}** missed readmissions")
            st.markdown(f"Each FN = ~$15,000 missed intervention opportunity")

        with col4:
            sec(f"📈 ROC Curve — {best_clf}")
            if hasattr(bm,"predict_proba"):
                proba_c   = bm.predict_proba(Xte_c)[:,1]
                fpr,tpr,_ = roc_curve(yc_te_s, proba_c)
                auc_val   = roc_auc_score(yc_te_s, proba_c)
                fig4, ax2 = plt.subplots(figsize=(5,4))
                ax2.plot(fpr,tpr,color=CLR["purple"],lw=2.5,label=f"AUC={auc_val:.4f}")
                ax2.plot([0,1],[0,1],color=CLR["grey"],ls="--")
                ax2.fill_between(fpr,tpr,alpha=0.1,color=CLR["purple"])
                ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
                ax2.set_title(f"ROC Curve — {best_clf}"); ax2.legend()
                plt.tight_layout(); st.pyplot(fig4); plt.close()

        insight("Recall is more important than Precision — missing a high-risk patient costs $15K+ in readmission.")
        warn(f"FN={fn:,} missed readmissions — each represents a preventable and costly hospital return.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔑 Tab 4 — Feature Importance")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        reg_df = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True) if S.get("reg_results") else pd.DataFrame()
        feats  = S["X_cols"]

        col1, col2 = st.columns(2)
        with col1:
            sec("🎯 Classification — Readmission Predictors")
            best_clf = clf_df.iloc[0]["Model"]
            bm = S["clf_models"][best_clf]
            if hasattr(bm,"feature_importances_"):
                imp = pd.DataFrame({"Feature":feats,"Importance":bm.feature_importances_})\
                        .sort_values("Importance",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(6,len(imp)*0.32)))
                colors_i = [CLR["purple"] if i>=len(imp)-5 else CLR["primary"]
                            for i in range(len(imp))]
                ax.barh(imp["Feature"], imp["Importance"], color=colors_i)
                ax.set_xlabel("Importance")
                ax.set_title(f"{best_clf} — Readmission Predictors")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            elif hasattr(bm,"coef_"):
                coef_vals = bm.coef_[0] if bm.coef_.ndim>1 else bm.coef_
                coef = pd.DataFrame({"Feature":feats,"Coef":np.abs(coef_vals)})\
                         .sort_values("Coef",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(6,len(coef)*0.32)))
                ax.barh(coef["Feature"], coef["Coef"], color=CLR["purple"])
                ax.set_xlabel("|Coefficient|"); ax.set_title(f"{best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                info(f"{best_clf} doesn't expose feature importances.")

        with col2:
            sec("📈 Regression — Length of Stay Predictors")
            if not reg_df.empty:
                best_reg = reg_df.iloc[0]["Model"]
                rm = S["reg_models"][best_reg]
                if hasattr(rm,"feature_importances_"):
                    imp2 = pd.DataFrame({"Feature":feats,"Importance":rm.feature_importances_})\
                             .sort_values("Importance",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(6,len(imp2)*0.32)))
                    ax2.barh(imp2["Feature"], imp2["Importance"], color=CLR["teal"])
                    ax2.set_xlabel("Importance")
                    ax2.set_title(f"{best_reg} — Stay Length Predictors")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
                elif hasattr(rm,"coef_"):
                    coef2 = pd.DataFrame({"Feature":feats,"Coef":np.abs(rm.coef_)})\
                              .sort_values("Coef",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(6,len(coef2)*0.32)))
                    ax2.barh(coef2["Feature"], coef2["Coef"], color=CLR["teal"])
                    ax2.set_xlabel("|Coefficient|"); ax2.set_title(f"{best_reg}")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
            else:
                info("Train regression models in Tab 1.")

        insight("number_inpatient, high_utilization, and total_visits expected as top readmission predictors.")
        insight("num_lab_procedures and num_medications typically top stay-length predictors.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICT — PATIENT RISK SCORER
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🔮 Tab 5 — Interactive Patient Readmission Risk Scorer")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        info("Enter patient profile to predict 30-day readmission risk.")

        col1, col2, col3 = st.columns(3)
        with col1:
            sec("👤 Patient Demographics")
            age_mid_val  = st.slider("Patient Age", 5, 95, 70, 5)
            gender_val   = st.selectbox("Gender", ["Female","Male"])
            race_val     = st.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"])

        with col2:
            sec("🏥 Admission Details")
            adm_type     = st.selectbox("Admission Type", [1,2,3], format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective"}[x])
            time_hosp    = st.slider("Days in Hospital", 1, 14, 4)
            num_diag     = st.slider("Number of Diagnoses", 1, 16, 5)
            num_lab      = st.slider("Lab Procedures", 1, 132, 40)
            num_proc     = st.slider("Procedures", 0, 6, 1)
            num_meds     = st.slider("Medications", 1, 81, 15)

        with col3:
            sec("📊 Utilization & Treatment")
            out_visits   = st.slider("Prior Outpatient Visits", 0, 20, 0)
            emg_visits   = st.slider("Prior Emergency Visits", 0, 20, 0)
            inp_visits   = st.slider("Prior Inpatient Visits", 0, 20, 1)
            insulin_val  = st.selectbox("Insulin", ["No","Steady","Up","Down"])
            diabetes_med = st.selectbox("On Diabetes Medication", ["Yes","No"])
            med_changed  = st.selectbox("Medication Changed", ["No","Ch"])

            # Computed flags
            total_visits_val  = out_visits + emg_visits + inp_visits
            q75_visits        = df['total_visits'].quantile(0.75)
            high_util_val     = 1 if total_visits_val > q75_visits else 0
            long_stay_val     = 1 if time_hosp >= 7 else 0
            ins_map           = {"No":0,"Steady":1,"Down":2,"Up":3}
            insulin_enc_val   = ins_map[insulin_val]
            uses_ins_val      = 1 if insulin_enc_val >= 1 else 0
            is_senior_val     = 1 if age_mid_val >= 65 else 0
            emg_adm_val       = 1 if adm_type == 1 else 0

            st.metric("Total Prior Visits",   f"{total_visits_val}")
            st.metric("High Utilization",     "⚠️ YES" if high_util_val else "✅ No")
            st.metric("Long Stay",            "⚠️ YES" if long_stay_val else "✅ No")
            st.metric("Senior Patient",       "⚠️ YES" if is_senior_val else "✅ No")

        st.markdown("---")
        if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
            def enc_cat(col, val):
                if col in df.columns and col+'_enc' in df.columns:
                    mapping = dict(zip(df[col].astype(str), df[col+'_enc']))
                    return int(mapping.get(str(val), 0))
                return 0

            input_row = pd.DataFrame([{
                "age_mid":                  age_mid_val,
                "num_lab_procedures":       num_lab,
                "num_procedures":           num_proc,
                "num_medications":          num_meds,
                "number_outpatient":        out_visits,
                "number_emergency":         emg_visits,
                "number_inpatient":         inp_visits,
                "number_diagnoses":         num_diag,
                "total_visits":             total_visits_val,
                "total_med_changes":        1 if med_changed=="Ch" else 0,
                "num_active_meds":          uses_ins_val,
                "high_utilization":         high_util_val,
                "long_stay":                long_stay_val,
                "uses_insulin":             uses_ins_val,
                "is_senior":                is_senior_val,
                "emergency_admission":      emg_adm_val,
                "race_enc":                 enc_cat("race", race_val),
                "gender_enc":               enc_cat("gender", gender_val),
                "max_glu_serum_enc":        0,
                "A1Cresult_enc":            0,
                "change_enc":               enc_cat("change", med_changed),
                "diabetesMed_enc":          enc_cat("diabetesMed", diabetes_med),
                "insulin_enc":              insulin_enc_val,
                "metformin_enc":            0,
                "glipizide_enc":            0,
                "glyburide_enc":            0,
                "pioglitazone_enc":         0,
                "rosiglitazone_enc":        0,
                "glimepiride_enc":          0,
                "admission_type_id":        adm_type,
                "discharge_disposition_id": 1,
                "admission_source_id":      1,
            }])

            input_aligned = pd.DataFrame([{k: float(input_row[k].iloc[0])
                                           if k in input_row.columns else 0.0
                                           for k in S["X_cols"]}])
            input_sc = S["scaler"].transform(input_aligned)

            sec("🎯 30-Day Readmission Risk — All Models")
            pred_rows = []
            for name, model in S["clf_models"].items():
                use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
                Xin    = input_sc if use_sc else input_aligned
                pred   = model.predict(Xin)[0]
                prob   = model.predict_proba(Xin)[0][1] if hasattr(model,"predict_proba") else None
                pred_rows.append({
                    "Model":       name,
                    "Prediction":  "🚨 HIGH RISK" if pred==1 else "✅ LOW RISK",
                    "Probability": f"{prob*100:.1f}%" if prob is not None else "N/A",
                })
            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df, use_container_width=True)

            high_risk_votes = sum(1 for r in pred_rows if "HIGH" in r["Prediction"])
            if high_risk_votes > len(pred_rows)/2:
                st.error(f"🚨 HIGH READMISSION RISK — {high_risk_votes}/{len(pred_rows)} models flag this patient. Activate intervention protocol NOW.")
            else:
                st.success(f"✅ Lower readmission risk — only {high_risk_votes}/{len(pred_rows)} models flag concern.")

            st.markdown("---")
            sec("📋 Risk Factor Analysis")
            risks = []
            if high_util_val:
                risks.append(("🔴","High utilization: total visits={} (>Q75={:.0f}) — strongest readmission predictor".format(total_visits_val, q75_visits)))
            if inp_visits >= 2:
                risks.append(("🔴",f"Prior inpatient visits: {inp_visits} — pattern of repeated hospitalizations"))
            if long_stay_val:
                risks.append(("🟡",f"Long stay: {time_hosp} days — serious complications during this admission"))
            if emg_adm_val:
                risks.append(("🟡","Emergency admission — unplanned, suggests unstable condition"))
            if uses_ins_val:
                risks.append(("🟡","Insulin-dependent — advanced diabetes, harder glucose control"))
            if is_senior_val:
                risks.append(("🟡",f"Senior patient ({age_mid_val} years) — multiple comorbidities, slower recovery"))
            if emg_visits >= 2:
                risks.append(("🔴",f"Prior emergency visits: {emg_visits} — frequent unplanned crises"))
            if med_changed == "Ch":
                risks.append(("🟡","Medications changed during admission — clinical instability signal"))

            if risks:
                for level, msg in risks:
                    color = "warn" if level in ["🔴","🟡"] else "info"
                    if level == "🔴":
                        st.markdown(f'<div class="warn-box"><p>🔴 {msg}</p></div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box"><p>🟡 {msg}</p></div>',
                                    unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box"><p>✅ No major risk factors detected — standard discharge protocol appropriate.</p></div>',
                            unsafe_allow_html=True)
