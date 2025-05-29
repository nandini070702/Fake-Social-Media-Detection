import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load processed features
data_path = "datasets/image_dataset/processed_faces"
X = np.load(os.path.join(data_path, "image_features.npy"))
y = np.load(os.path.join(data_path, "image_labels.npy"))

# Standardize the features (Same as during training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load models
rf_model = joblib.load("backend/ml/models/random_forest_image.pkl")
xgb_model = joblib.load("backend/ml/models/xgboost_image.pkl")

# Make predictions
rf_preds = rf_model.predict(X_scaled)
xgb_preds = xgb_model.predict(X_scaled)

# Evaluate
print(f"✅ Random Forest Accuracy: {accuracy_score(y, rf_preds):.4f}")
print(f"✅ XGBoost Accuracy: {accuracy_score(y, xgb_preds):.4f}")

print("\n✅ Classification Report for XGBoost:\n", classification_report(y, xgb_preds))
print("\n✅ Classification Report for Random Forest:\n", classification_report(y, rf_preds))
