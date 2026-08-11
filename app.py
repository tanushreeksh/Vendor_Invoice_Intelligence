import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# Page Configuration
st.set_page_config(
    page_title = "Vendor Invoice Intelligence Portal",
    page_icon = "📦",
    layout = "wide"
)


# Header section
st.markdown("""
# 📦Vendor Invoice Intelligence Portal

This internal analysis portal leverages machine learning to
- **Forecast Freight Costs accurately**
- **Detect Risky or Abnormal Vendor Invoices**
- **Reduce Finacial Leakage and Manual Workload**
"""
)

st.divider()


# Sidebar
st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Model",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact:**
- Improved Cost Forecasting
- Reduced Invoice Flaws and Anomalies
- Faster Finance Operations and Descision Making
""")



# Freight Cost Prediction
if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict Freight Cost for a Vendor Invoice using **Quantity** and **Dollars**
    to support budgeting, forecasting and vendor negotiations.
    """)

    with st.form("freight_cost_form"):
        quantity = st.number_input("Quantity", min_value=1, value=100)
        dollars = st.number_input("Dollars", min_value=0.0, value=1000.0, step=10.0)

        submitted = st.form_submit_button("Predict Freight Cost")

    if submitted:
        input_df = pd.DataFrame([{
            "Quantity": quantity,
            "Dollars": dollars,
        }])

        result_df = predict_freight_cost(input_df)
        prediction = max(result_df["Predicted_Freight"].iloc[0], 0)

        st.success(f"Predicted Freight Cost: ${prediction:,.2f}")

        with st.expander("Input features used"):
            st.dataframe(input_df)



# Invoice flag Prediction
else:
    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight, or delivery patterns.
""")

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)

        with col2:
            total_brands = st.number_input("Total Brands (on PO)", min_value=1, value=3)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)

        with col3:
            po_date = st.date_input("PO Date", key="flag_po_date")
            invoice_date = st.date_input("Invoice Date", key="flag_invoice_date")
            pay_date = st.date_input("Pay Date", key="flag_pay_date")

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

    if submit_flag:
        days_po_to_invoice = (invoice_date - po_date).days
        days_to_pay = (pay_date - invoice_date).days

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "Freight": [freight],
            "total_brands": [total_brands],
            "total_item_quantity": [total_item_quantity],
            "days_po_to_invoice": [days_po_to_invoice],
            "days_to_pay": [days_to_pay],
        }

        flag_prediction = predict_invoice_flag(input_data)['Predicted_Invoice_Flag']
        is_flagged = bool(flag_prediction.iloc[0])

        if is_flagged:
            st.error("Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("Invoice is **SAFE for Auto APPROVAL**")