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
    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict Freight Cost for a Vendor Invoice using **Shipment Details** and **Vendor Information**
    to support budgeting, forecasting and vendor negotiations.
    """)

    with st.form("freight_cost_form"):
        col1,col2 = st.columns(2)

        with col1:
            