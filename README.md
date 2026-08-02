# Vendor Invoice Analytics

Two ML models built on a procurement dataset — one predicts freight cost for vendor invoices, the other flags invoices that look off and probably need a manual review. Both are wired up to a Streamlit app so you can actually try them instead of just reading code.

## Why I built this

Freight cost is annoying to plan for ahead of time, and most teams don't have the bandwidth to manually check every single invoice for mismatches or weird patterns. I wanted to see if a fairly simple ML pipeline could help with both — one as a regression problem, one as classification — using the same underlying dataset.

## Project 1: Freight Cost Prediction

Predicts what freight cost should be for an invoice, based on order size, vendor history, and how long things took between PO, invoice, and payment.

**Features used:** Quantity, Dollars, price-per-unit, days from PO to invoice, days to pay, invoice month, vendor invoice frequency

I compared Linear Regression, Decision Tree, and Random Forest, and picked the winner using 5-fold cross-validation rather than a single train/test split (a single split can make a weaker model look better than it is just by luck).

**Result:** ~96% R² (explains 96% of the variance in freight cost), validated across folds.

**Where it struggles:** error is noticeably higher on small orders (roughly under $6,000). My guess is freight often has a flat/minimum charge that doesn't scale down for small orders the way the model assumes it should — haven't fully confirmed this, but the pattern held up across several test cases I checked manually.

## Project 2: Invoice Risk Flagging

Flags invoices that look risky — mainly cases where the invoice total doesn't line up with the underlying purchase order, or where something about the order pattern (brand count, quantity mismatch) looks unusual.

**Features used:** invoice quantity, freight, total brands per PO, total item quantity per PO, days from PO to invoice, days to pay

Used a Random Forest, tuned with GridSearchCV, with class-weighted training since risky invoices are the minority class.

**Result:** 84% accuracy, 67% recall on the risky class.

**A mistake I made and fixed:** the first version of this model included the exact columns used to build the risk label in the first place (`invoice_dollars` vs `total_item_dollars`). That meant the model wasn't really learning to spot risk — it was just reverse-engineering the rule I used to create the label, which is a classic data leakage problem. Removed those columns, rebuilt the feature set from things the model couldn't "cheat" with, and the accuracy dropped a bit — which is expected and honestly a sign the fix worked, not a step backward.

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
| Freight Cost Prediction | R² (cross-validated) | ~96% |
| Freight Cost Prediction | Median error | ~$12 |
| Invoice Risk Flagging | Accuracy | 84% |
| Invoice Risk Flagging | Recall (risky class) | 67% |

## Dataset

Vendor invoice / procurement dataset from Kaggle — includes vendor invoices, purchase orders, and line-item purchase records for a distribution business.
