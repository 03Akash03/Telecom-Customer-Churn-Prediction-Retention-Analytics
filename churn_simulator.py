# churn_simulator.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# 1. Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Churn Probability Simulator",
    layout="wide"
)

st.title("Churn Probability Simulator")
st.markdown(
    """
    Use this simulator to estimate the probability of a customer churning based on their profile
    and subscribed services. Adjust the inputs below to see the predicted churn probability.
    """
)

# Load pre-trained model
model = joblib.load("churn_model.pkl")

# Optional: load original dataset for default values
df = pd.read_csv("Dataset.csv")

# ------------------------------
# 2. User Inputs
# ------------------------------
with st.expander("Customer Profile and Subscription"):
    col1, col2 = st.columns(2)

    # ------------------------------
    # Column 1: Customer Info
    # ------------------------------
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 8700.0, 0.0)
        senior = st.checkbox("Senior Citizen")
        is_month_to_month = st.checkbox("Month-to-Month Contract", True)
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")

    # ------------------------------
    # Column 2: Services Info
    # ------------------------------
    with col2:
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.checkbox("Online Security")
        online_backup = st.checkbox("Online Backup")
        device_protection = st.checkbox("Device Protection")
        tech_support = st.checkbox("Tech Support")
        streaming_tv = st.checkbox("Streaming TV")
        streaming_movies = st.checkbox("Streaming Movies")
        paperless_billing = st.checkbox("Paperless Billing")
        payment_method = st.selectbox(
            "Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"]
        )

# ------------------------------
# 3. Derived Features
# ------------------------------

# Service Count
services_selected = {
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies
}
service_count = sum(services_selected.values())

# Display selected services
st.markdown(f"**Number of Services Selected:** {service_count}")
cols = st.columns(len(services_selected))
for i, (service, selected) in enumerate(services_selected.items()):
    if selected:
        cols[i].success(service)
    else:
        cols[i].write(service)

# Tenure Bucket
if tenure <= 6:
    tenure_bucket = "New"
elif tenure <= 24:
    tenure_bucket = "Standard"
else:
    tenure_bucket = "Loyal"

st.info(f"**Tenure Bucket:** {tenure_bucket} (based on {tenure} months)")

# Average Monthly Spend
avg_monthly_spend = total_charges / (tenure if tenure > 0 else 1)
st.metric("Average Monthly Spend ($)", f"{avg_monthly_spend:,.2f}")

# ------------------------------
# 4. Prepare Input DataFrame
# ------------------------------
input_dict = {
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "AvgMonthlySpend": [avg_monthly_spend],
    "ServiceCount": [service_count],
    "SeniorCitizen": [1 if senior else 0],
    "IsMonthToMonth": [1 if is_month_to_month else 0],
    "Partner": ["Yes" if partner else "No"],
    "Dependents": ["Yes" if dependents else "No"],
    "MultipleLines": ["No phone service"],  # default
    "InternetService": [internet_service],
    "OnlineSecurity": ["Yes" if online_security else "No"],
    "OnlineBackup": ["Yes" if online_backup else "No"],
    "DeviceProtection": ["Yes" if device_protection else "No"],
    "TechSupport": ["Yes" if tech_support else "No"],
    "StreamingTV": ["Yes" if streaming_tv else "No"],
    "StreamingMovies": ["Yes" if streaming_movies else "No"],
    "PaperlessBilling": ["Yes" if paperless_billing else "No"],
    "PaymentMethod": [payment_method],
    "TenureBucket": [tenure_bucket]
}
input_df = pd.DataFrame(input_dict)

# ------------------------------
# 5. Prediction
# ------------------------------

# Train dummy model if not fitted (demo only)
if not hasattr(model.named_steps['classifier'], "coef_"):
    st.info("Training demo model on dummy data for demonstration...")
    X_demo = pd.DataFrame({
        "tenure": np.random.randint(0, 72, 100),
        "MonthlyCharges": np.random.uniform(18, 120, 100),
        "TotalCharges": np.random.uniform(18, 120*72, 100),
        "AvgMonthlySpend": np.random.uniform(18, 120, 100),
        "ServiceCount": np.random.randint(0, 6, 100),
        "SeniorCitizen": np.random.randint(0, 2, 100),
        "IsMonthToMonth": np.random.randint(0, 2, 100),
        "Partner": np.random.choice(["Yes", "No"], 100),
        "Dependents": np.random.choice(["Yes", "No"], 100),
        "MultipleLines": np.random.choice(["Yes", "No phone service"], 100),
        "InternetService": np.random.choice(["No", "DSL", "Fiber optic"], 100),
        "OnlineSecurity": np.random.choice(["Yes", "No"], 100),
        "OnlineBackup": np.random.choice(["Yes", "No"], 100),
        "DeviceProtection": np.random.choice(["Yes", "No"], 100),
        "TechSupport": np.random.choice(["Yes", "No"], 100),
        "StreamingTV": np.random.choice(["Yes", "No"], 100),
        "StreamingMovies": np.random.choice(["Yes", "No"], 100),
        "PaperlessBilling": np.random.choice(["Yes", "No"], 100),
        "PaymentMethod": np.random.choice(["Bank transfer", "Credit card", "Electronic check", "Mailed check"], 100),
        "TenureBucket": np.random.choice(["New", "Standard", "Loyal"], 100)
    })
    y_demo = np.random.randint(0, 2, 100)
    model.fit(X_demo, y_demo)

churn_prob = model.predict_proba(input_df)[:, 1][0]

# ------------------------------
# 6. Display Results
# ------------------------------
st.subheader("Churn Probability")
st.metric(label="Probability of Churn", value=f"{churn_prob*100:.2f}%")
