# ==============================
# Aegis Project: Real-time Transaction Fraud Detection
# Fully functional Jupyter Notebook
# ==============================

# Step 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("All libraries imported successfully!")

# ------------------------------
# Step 2+3: Load Dataset + Feature Engineering
# ------------------------------
df = pd.read_csv("transactions.csv")
print("\nDataset loaded successfully!")
print(df.head())

# Define numeric features and labels
features = ['amount', 'latitude', 'longitude']
X = df[features]
y = df['label']
print("\nFeatures and labels defined successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------
# Step 4: Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain/test split done.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ------------------------------
# Step 5: Train Isolation Forest Model
# ------------------------------
iso_model = IsolationForest(contamination=0.05, random_state=42)
iso_model.fit(X_train)

# Predict anomalies on test set (-1 = anomaly, 1 = normal)
y_pred = iso_model.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]  # Convert to 1=fraud, 0=normal
print("\nModel training completed.")

# ------------------------------
# Step 6: Evaluate Model
# ------------------------------
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))  # zero_division avoids warnings

# ------------------------------
# Step 7: Simulate Real-Time Transactions
# ------------------------------
print("\nSimulating real-time transactions:")
new_transactions = pd.DataFrame([
    {'amount': 1200, 'latitude': 12.9716, 'longitude': 77.5946},
    {'amount': 5000, 'latitude': 40.7128, 'longitude': -74.0060},
    {'amount': 300, 'latitude': 19.0760, 'longitude': 72.8777},
])

for i, tx in new_transactions.iterrows():
    tx_df = pd.DataFrame([tx[features]])  # Ensure DataFrame with correct columns
    score = iso_model.predict(tx_df)[0]
    label = "Fraud" if score == -1 else "Not Fraud"
    print(f"Transaction {i+1}: {tx.to_dict()} -> {label}")

# ------------------------------
# Step 8: Save / Load Model + Test Prediction
# ------------------------------
joblib.dump(iso_model, "aegis_model.joblib")
print("\nModel saved as 'aegis_model.joblib'.")

loaded_model = joblib.load("aegis_model.joblib")
print("Model loaded successfully!")

# Test the loaded model
sample_tx = pd.DataFrame([{'amount': 2000, 'latitude': 28.7041, 'longitude': 77.1025}])
score = loaded_model.predict(sample_tx)[0]
label = "Fraud" if score == -1 else "Not Fraud"
print("Sample transaction prediction:", label)

# ------------------------------
# Step 9: Visualization (Matplotlib Only)
# ------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['latitude'], df['longitude'], c=df['label'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Transaction Locations: Fraud vs Legitimate")
plt.colorbar(label='Fraud Label (0=Normal, 1=Fraud)')
plt.show()

# ------------------------------
# Step 10: Optional: Real-Time Streaming Simulation
# ------------------------------
print("\nOptional real-time streaming simulation:")
stream_transactions = pd.DataFrame([
    {'amount': 700, 'latitude': 13.0827, 'longitude': 80.2707},
    {'amount': 2500, 'latitude': 34.0522, 'longitude': -118.2437}
])

for i, tx in stream_transactions.iterrows():
    tx_df = pd.DataFrame([tx[features]])
    score = loaded_model.predict(tx_df)[0]
    label = "Fraud" if score == -1 else "Not Fraud"
    print(f"Streamed Transaction {i+1}: {tx.to_dict()} -> {label}")
