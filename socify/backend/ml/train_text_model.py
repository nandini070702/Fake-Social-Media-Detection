import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib
import os

# Load the dataset
data_path = "datasets/text_dataset/bot_detection_data.csv"
df = pd.read_csv(data_path)

# Show first few rows and columns to check data
print(df.head())
print("\nColumns:", df.columns)

# Drop rows with missing values
df.dropna(inplace=True)

# Show class distribution
print("\nClass distribution:\n", df['Bot Label'].value_counts(normalize=True).rename("proportion"))

# Set target and text columns
target_column = 'Bot Label'
text_column = 'Tweet'

# Separate target
y = df[target_column]

# Drop non-numeric columns to keep only numeric features
non_numeric_cols = ['User ID', 'Username', 'Location', 'Created At', 'Hashtags', text_column]
X = df.drop(columns=non_numeric_cols + [target_column], errors='ignore')

# Optionally encode 'Verified' if it's boolean/string
if 'Verified' in X.columns:
    X['Verified'] = X['Verified'].astype(int)

# Print remaining columns for verification
print("\nNumeric feature columns used for training:\n", X.columns)

# Function to remove correlated features
def remove_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"\nDropping correlated features: {to_drop}")
    return X.drop(columns=to_drop)

X = remove_correlated_features(X)

# Feature selection using Mutual Information
selector = SelectKBest(mutual_info_classif, k=min(10, X.shape[1]))  # handle case when <10 features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

print("\nSelected top features:", list(selected_features))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_resampled, y_resampled)

# Evaluate
print("\nRandom Forest Report:")
print(classification_report(y_resampled, rf_model.predict(X_resampled)))

print("\nXGBoost Report:")
print(classification_report(y_resampled, xgb_model.predict(X_resampled)))

# Save models and preprocessing objects
os.makedirs("backend/models", exist_ok=True)
joblib.dump(rf_model, "backend/models/rf_model.pkl")
joblib.dump(xgb_model, "backend/models/xgb_model.pkl")
joblib.dump(scaler, "backend/models/text_scaler.pkl")
joblib.dump(selector, "backend/models/text_selector.pkl")

print("\nModels and preprocessors saved successfully.")
