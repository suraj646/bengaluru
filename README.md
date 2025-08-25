# 🏠 House Price Prediction  

A Streamlit web app that predicts **house prices** based on location, area type, square footage, number of bathrooms, and BHK.  
The model is trained using **XGBoost**, with feature/target scaling and target transformation applied.  

## 🚀 Live Demo

You can try the deployed app here:  
👉 [Bengaluru House Price Prediction](https://bengaluru-laihsqzxenhrowfxj4pzy6.streamlit.app/)
---

## 🚀 Features
- Predicts house prices using an ML model (`xgb_model.pkl`)  
- Encodes **location** and **area type** with mean-target encoding  
- Handles feature scaling (`scaler_X.pkl`) and target scaling (`scaler_y.pkl`)  
- User-friendly **Streamlit UI**  
- Debugging outputs for checking model behavior  

---

## 📂 Project Structure
- xgb_model.pkl # Trained XGBoost model
- scaler_X.pkl # Feature scaler
- scaler_y.pkl # Target scaler
- location_encoding.csv # Encodings for locations
- area_encoding.csv # Encodings for area types
- app.py # Streamlit app (main file)
- README.md # Project description (this file)

📊 Model Training (Quick Overview)

Algorithm: XGBoost Regressor

Feature Engineering:

Square Feet

Bathrooms

BHK

Encoded Area Type

Encoded Location

Target transformation: log1p(price)

Scaling applied on both features and target

🎮 Usage

Select location from dropdown

Select area type

Enter square feet, bathrooms, and BHK

Click Predict Price → App shows estimated house price

🐞 Debug Mode

The app also prints:

Input before scaling

Input after scaling

Model raw prediction

Inverse-scaled prediction

This helps troubleshoot issues when predictions don’t look realistic.

📸 Screenshot

<img width="1285" height="951" alt="image" src="https://github.com/user-attachments/assets/137d797d-fdd6-4a15-afca-b790ad70c25c" />

<img width="1189" height="809" alt="image" src="https://github.com/user-attachments/assets/e721c945-6247-41b7-b9f2-3796480c2f3d" />

📌 Requirements

Python 3.8+

Libraries:

streamlit

pandas

numpy

xgboost

scikit-learn

joblib

👨‍💻 Author

Developed by [Suraj]
