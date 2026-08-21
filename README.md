# Vendor Invoice Analytics

This project implements an **end-to-end machine learning system** designed to support finance teams by,
1. **Predicting expected freight cost** for vendor invoices
2. **Flagging high-risk invoices** that requires manual review due to abnormal cost, freight or operational patterns.

## Why I built this

Freight cost is annoying to plan for ahead of time, and most teams don't have the bandwidth to manually check every single invoice for mismatches or weird patterns. I wanted to see if a fairly simple ML pipeline could help with both using the same underlying dataset.

<h2><a class = 'anchor' id = 'project-overview'></a>Business Objectives</h2>

## 1: Freight Cost Prediction (Regression)
Objective - Predict the expected freight cost for a vendor invoice using quantity, invoice value and historical behavior.

WHY it matters?
1. Poor freight estimation impacts margin analysis and budgeting
2. Early prediction improves decision making and vendor negotiation

**Features used:** Quantity, Dollars

## 2: Invoice Risk Flagging (Classification)

Objective - Identify invoices that may require review based on mismatches and unusual purchase order patterns.

WHY it matters?
1. Early risk detection helps reduce billing errors
2. Prevent financial discrepancies and prioritize invoices for manual review.

**Features used:** invoice quantity, freight, PO-to-invoice delay, and total item quantity."

Prediction uses two probability thresholds rather than a single 0.5 cutoff: invoices scoring above 0.40 are flagged risky, below 0.30 are flagged not risky and anything in between is routed to manual review instead of forced into either bucket.

## Tech stack

Python, SQL (SQLite), scikit-learn, pandas, NumPy, Streamlit


## Running it

```bash
python freight_cost_prediction/train.py
python invoice_flagging/train.py
streamlit run app.py
```

## Results at a glance

| Model | Metric | Score |
|---|---|---|
| Freight Cost Prediction | R² (cross-validated) | 96% |
| Freight Cost Prediction | Median error | $2.75 |
| Invoice Risk Flagging | Accuracy | 84% |
| Invoice Risk Flagging | Recall (risky class 0.40 threshold) | 71% |

## Dataset

Vendor invoice / procurement dataset from Kaggle — includes vendor invoices, purchase orders, and line-item purchase records for a distribution business.
