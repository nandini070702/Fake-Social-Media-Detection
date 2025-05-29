import numpy as np
import joblib
from merge_predictions import merge_predictions
from sklearn.feature_selection import SelectKBest, chi2

# Load trained text models
text_xgb = joblib.load("backend/ml/models/xgboost_text.pkl")
text_rf = joblib.load("backend/ml/models/random_forest_text.pkl")

# Load processed text and image datasets
text_features = np.load("datasets/text_dataset/processed_bot_data.npy")  # Ensure correct path
image_features = np.load("datasets/image_dataset/processed_faces/image_features.npy")

# ✅ Apply the same feature selection used during training (Select Top 500 Features)
selector = SelectKBest(chi2, k=500)
text_features_selected = selector.fit_transform(text_features, np.zeros(text_features.shape[0]))  # Use dummy labels

# Ensure we don't exceed dataset sizes
max_samples = min(len(text_features_selected), len(image_features))  # Get the smallest dataset size
random_indices = np.random.choice(max_samples, size=5, replace=False)  # Pick 5 valid indices

for idx in random_indices:
    text_sample = text_features_selected[idx].reshape(1, -1)  # ✅ Fix: Use 500 selected features
    image_sample = image_features[idx].reshape(1, -1)

    # Get final classification
    final_prediction = merge_predictions(text_sample, image_sample)
    
    print(f"✅ Profile {idx + 1}: {'REAL' if final_prediction[0] == 1 else 'FAKE'}")
