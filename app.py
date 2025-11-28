# ===============================
# Aegis Fraud Detection Streamlit UI
# ===============================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Load model and dataset
# -----------------------
@st.cache_resource
def load_model(path="aegis_model.joblib"):
    return joblib.load(path)

@st.cache_data
def load_data(path="transactions.csv"):
    return pd.read_csv(path)

model = load_model()
features = ['amount', 'latitude', 'longitude']
df = load_data()

# -----------------------
# Streamlit Page
# -----------------------
st.set_page_config(page_title="Aegis Fraud Detector", layout="centered")
st.title("Aegis — Real-time Transaction Fraud Detection")
st.write("Select a dataset row or enter custom transaction details to predict fraud.")

# -----------------------
# Dataset preview
# -----------------------
with st.expander("Preview Dataset"):
    st.dataframe(df.head(10))

# -----------------------
# Predict from selected row
# -----------------------
st.subheader("Select a row from the dataset")
row_index = st.number_input("Row index (0-based)", min_value=0, max_value=len(df)-1, value=0, step=1)
selected_row = df.iloc[int(row_index)]
st.write("Selected transaction:")
st.json(selected_row.to_dict())

def predict_from_df(input_df):
    pred = model.predict(input_df[features])
    labels = ["Fraud" if p == -1 else "Not Fraud" for p in pred]
    return labels

if st.button("Predict selected row"):
    tx_df = pd.DataFrame([selected_row[features]])
    label = predict_from_df(tx_df)[0]
    st.success(f"Prediction: {label}")

# -----------------------
# Predict custom transaction
# -----------------------
st.subheader("Or enter custom transaction details")
with st.form("custom_tx"):
    amount = st.number_input("Amount", min_value=0.0, value=float(df['amount'].median()))
    latitude = st.number_input("Latitude", value=float(df['latitude'].median()))
    longitude = st.number_input("Longitude", value=float(df['longitude'].median()))
    submitted = st.form_submit_button("Predict custom transaction")

if submitted:
    custom_df = pd.DataFrame([{'amount': amount, 'latitude': latitude, 'longitude': longitude}])
    custom_label = predict_from_df(custom_df)[0]
    if custom_label == "Fraud":
        st.error(f"Prediction: {custom_label}")
    else:
        st.success(f"Prediction: {custom_label}")

# -----------------------
# Visualizations
# -----------------------
st.subheader("Transaction Amount Distribution")
st.bar_chart(df['amount'])

st.subheader("Fraud vs Non-Fraud Count")
st.bar_chart(df['label'].value_counts())

st.subheader("Transaction Locations")
try:
    map_df = df[['latitude', 'longitude']].dropna()
    map_df = map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    st.map(map_df)
except Exception as e:
    st.write("Map unavailable:", e)

st.write("---")
st.write("Model: IsolationForest | Saved as `aegis_model.joblib`")
