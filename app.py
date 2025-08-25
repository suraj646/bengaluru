import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and scalers
# model = joblib.load("xgb_model.pkl")
# scaler_X = joblib.load("scaler_X.pkl")   # feature scaler
# scaler_y = joblib.load("scaler_y.pkl")   # target scaler
model = pickle.load(open('xgb_model.pkl', 'rb'))
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
# Load encodings from CSV
loc_price_mean = pd.read_csv("location_encoding.csv", index_col=0).squeeze("columns")
area_price_mean = pd.read_csv("area_encoding.csv", index_col=0).squeeze("columns")

# UI
st.title("🏠 House Price Prediction (with Debugging)")

# Location input
location = st.selectbox("Select Location", loc_price_mean.index)
location_enc = loc_price_mean[location]

# Area type input
area_type = st.selectbox("Select Area Type", area_price_mean.index)
area_type_enc = area_price_mean[area_type]

# Other numeric inputs
sqft = st.number_input("Enter Total Square Feet", min_value=500, max_value=10000, step=50)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    # Prepare input (⚠️ order must match training!)
    input_data = np.array([[bhk, sqft, bath, location_enc, area_type_enc]])

    # Debug: show raw input
    st.write("🔹 Input before scaling:", input_data)

    # Apply scaling (must match training pipeline)
    try:
        input_scaled = scaler_X.transform(input_data)
    except Exception as e:
        st.error(f"Scaler_X error: {e}")
        st.stop()

    st.write("🔹 Input after scaling:", input_scaled)

    # Predict (scaled)
    pred_scaled = model.predict(input_scaled).reshape(-1, 1)
    st.write("🔹 Raw model prediction (scaled):", pred_scaled)

    # Inverse scale output
    try:
        pred_inverse = scaler_y.inverse_transform(pred_scaled)[0][0]
    except Exception as e:
        st.error(f"Scaler_Y error: {e}")
        st.stop()

    st.write("🔹 After inverse scaling:", pred_inverse)

    # If you used log1p(price) during training, uncomment this:
    prediction = np.expm1(pred_inverse)

    # Show final prediction
    st.success(f"💰 Estimated Price: ₹ {prediction:,.2f} lakhs")

