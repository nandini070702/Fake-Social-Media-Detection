import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

# Load processed dataset
data_path = "datasets/text_dataset/processed_bot_data.csv"
print(f"Loading processed dataset from: {data_path}")
df = pd.read_csv(data_path)

# Load trained models
rf_model = joblib.load("backend/ml/models/random_forest_text.pkl")
xgb_model = joblib.load("backend/ml/models/xgboost_text.pkl")

# Separate features and labels
X = df.drop(columns=["Bot Label"])
y = df["Bot Label"]

# ✅ Apply the same scaling used during training
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Apply the same feature selection used during training (500 features)
selector = SelectKBest(chi2, k=500)
X_selected = selector.fit_transform(X_scaled, y)

# ✅ Make predictions using trained models
rf_preds = rf_model.predict(X_selected)
xgb_preds = xgb_model.predict(X_selected)

# ✅ Evaluate accuracy
print("Random Forest Accuracy:", accuracy_score(y, rf_preds))
print("XGBoost Accuracy:", accuracy_score(y, xgb_preds))

# ✅ Display classification report
print("\nClassification Report for XGBoost:\n", classification_report(y, xgb_preds))
