import numpy as np
import joblib
from merge_predictions import merge_predictions

# ✅ Load trained models
text_xgb = joblib.load("backend/ml/models/xgboost_text.pkl")
text_rf = joblib.load("backend/ml/models/random_forest_text.pkl")
image_xgb = joblib.load("backend/ml/models/xgboost_image.pkl")
image_rf = joblib.load("backend/ml/models/random_forest_image.pkl")

# ✅ Load feature selector to apply the same transformation as training
text_selector = joblib.load("backend/ml/models/feature_selector.pkl")

# ✅ Load dataset features & labels
text_features = np.load("datasets/text_dataset/processed_bot_data.npy")
image_features = np.load("datasets/image_dataset/processed_faces/image_features.npy")
true_labels = np.load("datasets/image_dataset/processed_faces/image_labels.npy")

# ✅ Ensure dataset sizes match
num_samples = min(len(text_features), len(image_features), len(true_labels))
text_features = text_features[:num_samples]
image_features = image_features[:num_samples]
true_labels = true_labels[:num_samples]

# ✅ Apply the same feature selection process as during training
text_features_selected = text_selector.transform(text_features)

# ✅ Make predictions
pred_text_xgb = text_xgb.predict(text_features_selected)
pred_text_rf = text_rf.predict(text_features_selected)
pred_image_xgb = image_xgb.predict(image_features)
pred_image_rf = image_rf.predict(image_features)
pred_merged = merge_predictions(text_features_selected, image_features)

# ✅ Save predictions as .npy files
np.save("backend/ml/results/text_xgb_predictions.npy", pred_text_xgb)
np.save("backend/ml/results/text_rf_predictions.npy", pred_text_rf)
np.save("backend/ml/results/image_xgb_predictions.npy", pred_image_xgb)
np.save("backend/ml/results/image_rf_predictions.npy", pred_image_rf)
np.save("backend/ml/results/merged_predictions.npy", pred_merged)

print("✅ Model predictions generated and saved successfully!")
