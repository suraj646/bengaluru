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

Developed by [Your Name]
