# Payment Probability Prediction

> Predicting customer payment likelihood from 6-month loan recovery data using machine learning.

---

## Overview

This project builds a binary classifier to predict whether a customer in a loan recovery portfolio will make a payment in a given month. The dataset covers **348,151 records** across a 6-month window (Feb 2023 – Jul 2023), tracking loan and payment history for customers in recovery.

The work is structured in four stages: **Data Cleaning → Exploratory Data Analysis → Modelling → Conclusions**.

---

## Repository Structure

```
├── Payment_Probability_Analysis.ipynb   # Main analysis notebook
├── Data_6Months.csv                     # Raw dataset (6-month loan recovery data)
├── Payment_Probability_Prediction.pptx  # Summary slide deck
├── Assigment_details.docx               # Original assignment brief
└── README.md
```

---

## Dataset

**File:** `Data_6Months.csv`  
**Shape:** 348,151 rows × 27 columns  
**Period:** February 2023 – July 2023

### Key Columns

| Column | Description |
|---|---|
| `Account_number` | Unique loan account identifier |
| `Principal_outstanding` | Remaining principal on the loan |
| `dpd_days` | Days past due |
| `dpd_bucket` | DPD category (`NCL` = Non-Current Loan, `GWO` = Gross Write-Off) |
| `prod_name` | Loan product type (`PL_SAL`, `PL_SELF`) |
| `disb_amt` | Disbursed loan amount |
| `tenor` | Loan tenure (months) |
| `roi` | Rate of interest |
| `emi` | Monthly EMI amount |
| `cibil_score` | Credit bureau score |
| `residence_state` | Customer's state of residence |
| `gwo_amt` | Gross write-off amount |
| `total_outstanding` | Total outstanding amount |
| `Pmt_amount` | Payment made in that month (target signal) |
| `month_end_date` | Month-end date for the record |

---

## Methodology

### 1. Data Cleaning

- Parsed mixed-format date columns (`disbursal_date`, `writeoff_date`, `month_end_date`)
- Cast `total_outstanding` from string to float
- Dropped columns with >90% missing values (`emp_type`, `industry_type`, `total_work_exp_months`)
- Imputed remaining numeric nulls with median; categorical nulls with `'Unknown'`
- Dropped rows missing `Account_number`

**Data Quality Highlights:**

| Issue | Resolution |
|---|---|
| `emp_type` / `industry_type` ~95% null | Dropped |
| `total_outstanding` stored as string | Cast to `float64` |
| Mixed date formats | Parsed with `dayfirst=True` |
| `cibil_score` 3.1% null | Median imputation |
| `Pmt_amount` 0.3% null | Treated as 0 (no payment) |

### 2. Exploratory Data Analysis

**Account Distribution:**
- 97.5% of accounts are NCL (Non-Current Loan); only 2.5% are GWO (Gross Write-Off)
- Salaried loans (`PL_SAL`) outnumber self-employed (`PL_SELF`) nearly 3:1
- Maharashtra, Tamil Nadu, and Uttar Pradesh hold the largest delinquent portfolios

**Payment Trends (Feb–Jul 2023):**
- Collections peaked in February (₹149M), dipped in April (₹91M), recovered to ₹127M by July
- Unique payer rate declined from 10.5% → 8.5% over the period, indicating worsening recovery

### 3. Modelling

**Target variable:** `paid = 1` if `Pmt_amount > 0`, else `0`  
**Payment rate:** 8.8% (class imbalance ≈ 9:1)

**Feature Engineering:**
- `months_since_disbursal` — derived from `disbursal_date` and `month_end_date`
- `months_since_writeoff` — derived from `writeoff_date` and `month_end_date`
- `dpd_bucket_enc` — binary encoding (GWO=1, NCL=0)
- `prod_enc` — label-encoded product name
- `state_enc` — top-15 states label-encoded, rest as `'Other'`

**Train/Test Split:** 80/20 stratified split (Train: 278,520 | Test: 69,631 rows)

**Class Imbalance:** Handled via `class_weight='balanced'` on all models

### 4. Model Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.8050 |
| Decision Tree (depth=6) | 0.8329 |
| **Random Forest (depth=8, 100 trees)** | **0.8485 ★** |

**Top Predictors (Random Forest):** `dpd_days`, `principal_outstanding`, `gwo_amt`

---

## Key Findings

1. The portfolio is heavily NCL-dominated (97.5%); GWO accounts represent a small but distinct segment
2. Monthly collections fell 39% from Feb to Apr before partially recovering
3. The payer rate trended down over 6 months — a signal of recovery challenges
4. Random Forest achieved the best ROC-AUC of **0.8485**, outperforming both the logistic regression baseline and the decision tree
5. Days past due, outstanding principal, and write-off amount are the strongest predictors of payment

---

## Next Steps

- Add rolling 3-month payment history as features for temporal signal
- Experiment with XGBoost / LightGBM for further AUC improvement
- Tune classification threshold to optimise precision vs. recall for the business case
- Enrich with full CIBIL bureau history to reduce data sparsity
- Build a monthly retraining pipeline as new recovery data is collected

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

```bash
jupyter notebook Payment_Probability_Analysis.ipynb
```

Make sure `Data_6Months.csv` is in the same directory as the notebook before running.

---

## Deliverables

| File | Description |
|---|---|
| `Payment_Probability_Analysis.ipynb` | Full analysis: cleaning, EDA, modelling, conclusions |
| `Payment_Probability_Prediction.pptx` | Executive summary slide deck |
| `Data_6Months.csv` | Raw input dataset |
