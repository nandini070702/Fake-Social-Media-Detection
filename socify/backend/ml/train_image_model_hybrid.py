import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load processed image features
data_path = "datasets/image_dataset/processed_faces"
X = np.load(os.path.join(data_path, "image_features.npy"))
y = np.load(os.path.join(data_path, "image_labels.npy"))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using oversampling
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_scaled, y)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train Random Forest (Better Hyperparameters)
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=50, min_samples_split=3, min_samples_leaf=2,
    max_features='sqrt', random_state=42
)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)

# Train XGBoost (Better Hyperparameters)
xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=15, colsample_bytree=0.95,
    subsample=0.95, scale_pos_weight=1.0, gamma=0.2, reg_lambda=2,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_accuracy = xgb_model.score(X_test, y_test)

# Save models
joblib.dump(rf_model, "backend/ml/models/random_forest_image.pkl")
joblib.dump(xgb_model, "backend/ml/models/xgboost_image.pkl")

print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"✅ XGBoost Accuracy: {xgb_accuracy:.4f}")
print("✅ Models saved successfully!")

# Save the scaler for use during inference
joblib.dump(scaler, "backend/ml/models/image_scaler.pkl")

# Ensemble logic (optional function to reuse during backend inference)
def predict_image_ensemble(rf_model, xgb_model, X_input):
    rf_prob = rf_model.predict_proba(X_input)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_input)[:, 1]
    ensemble_prob = (rf_prob + xgb_prob) / 2
    return (ensemble_prob > 0.5).astype(int), ensemble_prob

print("✅ Scaler saved!")
