# 🏥 P13 — Hospital Readmission Prediction (Diabetes)
**M3 · ML Engine Portfolio · Project 13 · Extra**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-UCI_Repository-0052CC)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## 📌 Project Overview

End-to-end predictive analytics on **101,763 diabetic patient encounters** from 130 US hospitals (1999–2008). Predicts which patients will be readmitted within 30 days of discharge — a key CMS quality metric and major cost driver for healthcare systems.

**Core Questions:**
- Which patient characteristics predict 30-day readmission?
- Do high healthcare utilizers have significantly higher readmission rates?
- Which medications and lab results are associated with readmission?
- What is the ROI of targeted post-discharge interventions?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI ML Repository · Diabetes 130-US Hospitals |
| Raw Records | 101,766 encounters |
| After Cleaning | 101,763 patients · 82 features |
| Hospitals | 130 US hospitals |
| Period | 1999–2008 |
| Null Marker | `?` used → loaded with `na_values='?'` |

### ⚠️ Columns Dropped
| Column | Reason |
|--------|--------|
| `weight` | 96.9% missing |
| `medical_specialty` | 49.1% missing |
| `payer_code` | 39.6% missing |
| `encounter_id`, `patient_nbr` | Identifiers only |
| `diag_1`, `diag_2`, `diag_3` | ICD codes — too granular |

### ⚠️ Key Encodings
- `max_glu_serum` — 94.7% missing → filled `'None'` (no test ordered)
- `A1Cresult` — 83.3% missing → filled `'None'` (no test ordered)
- `age` — range strings `[70-80)` → midpoint numeric (75)

---

## 🎯 Targets

| Type | Column | Description |
|------|--------|-------------|
| **Regression** | `time_in_hospital` | Days in hospital (1–14) |
| **Classification** | `readmitted_30` | 1 if readmitted within 30 days |

**Balance:** 88.8% not readmitted / 11.2% readmitted → **SEVERE IMBALANCE**

`class_weight='balanced'` **MANDATORY** on ALL classifiers

Evaluate with **F1, Recall, ROC-AUC** — NEVER accuracy

---

## ⚙️ Feature Engineering

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| `age_mid` | Age band midpoint | Ordinal age for ML |
| `total_visits` | outpatient+emergency+inpatient | Overall utilization history |
| `total_med_changes` | Count of Up+Down changes | Medication instability |
| `num_active_meds` | Count of Steady+Up meds | Medication burden |
| `high_utilization` | total_visits > Q75 | **18.93% vs 9.72% readmit ★** |
| `long_stay` | time_in_hospital ≥ 7 | Serious complications |
| `uses_insulin` | insulin_enc ≥ 1 | Advanced diabetes |
| `is_senior` | age_mid ≥ 65 | Multiple comorbidities |
| `emergency_admission` | admission_type_id == 1 | Unplanned deterioration |
| `readmitted_30` | readmitted == '<30' | CLF target |

---

## 📊 Key Findings

| Finding | Value | Insight |
|---------|-------|---------|
| Overall readmission rate | 11.2% | 11,357 preventable readmissions |
| High utilization readmit | 18.93% | **2× the average rate** ★ |
| Normal utilization readmit | 9.72% | Baseline |
| Long stay readmit | 13.23% | +2.6 pp above average |
| Insulin users readmit | 12.14% | Advanced disease signal |
| Senior patients readmit | 11.61% | Multiple comorbidities |
| Mean age | 66.0 years | Elderly population |
| Mean stay | 4.40 days | Range 1–14 |

---

## 📊 EDA Dashboard — 13 Tabs

| Tab | Title | Highlight |
|-----|-------|-----------|
| 1 | Data Overview | Shape, types, stats, dictionary |
| 2 | Readmission Analysis ★ | 30d vs >30d vs No · by admission type |
| 3 | Demographics ★ | Age, gender, race readmission rates |
| 4 | Hospital Utilization ★ | High utilization — 2× readmit rate |
| 5 | Medication Analysis ★ | 23 medications + insulin deep-dive |
| 6 | Lab Results ★ | Glucose serum + A1C analysis |
| 7 | Multicollinearity | VIF analysis |
| 8 | Correlation | Heatmap + top readmission predictors |
| 9 | Business KPIs ★ | Cost of readmissions · intervention ROI |
| 10 | Category Deep-Dive ★ | Age × Insulin × Race heatmaps |
| 11 | Statistical Tests ★ | T1–T4: utilization · age · stay · meds |
| 12 | Feature Engineering | Clinical logic behind each feature |
| 13 | Insights & Report | Report + Risk Segments + Med Impact download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons |
| 2 | Regression Results — R², MAE, RMSE · predict stay length |
| 3 | Classification Results — F1, Recall, ROC-AUC · predict readmission |
| 4 | Feature Importance — top readmission risk predictors |
| 5 | Interactive Patient Risk Scorer — real-time readmission probability |

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 🔴 Critical | 48-hour post-discharge call for all high-utilization patients |
| 🔴 Critical | Pharmacist review for patients with ≥3 medication changes |
| 🟡 High | Enhanced discharge planning for all patients ≥65 |
| 🟡 High | Diabetes educator consultation for all insulin-dependent patients |
| 🟡 High | 72-hour follow-up for all emergency admissions |
| 🟢 Medium | Deploy ML readmission risk score at point of admission |

---

## 🗂 Project Structure

```
📁 Repo_13_Hospital_Readmission/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── readmission_clean.csv       ← from P13_clean_data.py (Jupyter)
└── pages/
    ├── EDA_dashboard.py             ← 13-tab analysis
    └── ML_Models.py                 ← 5-tab ML engine
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_13_Hospital_Readmission.git
cd Repo_13_Hospital_Readmission
pip install -r requirements.txt

# Step 1: Generate clean dataset
# Run P13_clean_data.py in Jupyter → saves readmission_clean.csv
# Copy to data/ via File Explorer (NEVER open in Excel)

streamlit run Home.py
```

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 13 End-to-End Data Science Projects**
