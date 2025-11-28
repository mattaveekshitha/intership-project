# ===============================
# NEW FEATURE: Select Row to Predict
# ===============================

print("\nSelect a row number from the dataset to detect fraud:")
print(f"Valid rows: 0 to {len(df)-1}")

row_number = int(input("Enter row number: "))

if 0 <= row_number < len(df):
    selected_row = df.iloc[row_number]
    print("\nSelected row:")
    print(selected_row)

    tx_df = pd.DataFrame([{
        'amount': selected_row['amount'],
        'latitude': selected_row['latitude'],
        'longitude': selected_row['longitude']
    }])

    score = loaded_model.predict(tx_df)[0]
    label = "Fraud" if score == -1 else "Not Fraud"

    print("\nPrediction for selected row:", label)
else:
    print("Invalid row number!")
