from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
import numpy as np

# Load processed text dataset
text_features = np.load("datasets/text_dataset/processed_bot_data.npy")
true_labels = np.load("datasets/image_dataset/processed_faces/image_labels.npy")

# ✅ Match dataset sizes
num_samples = min(len(text_features), len(true_labels))
text_features = text_features[:num_samples]
true_labels = true_labels[:num_samples]

# ✅ Improved: Use mutual_info_classif instead of chi2 for better feature selection
# mutual_info_classif captures non-linear relationships better than chi2
selector = SelectKBest(mutual_info_classif, k=500)
text_features_selected = selector.fit_transform(text_features, true_labels)

# ✅ Save processed features
np.save("datasets/text_dataset/processed_text_features.npy", text_features_selected)
# ✅ Save the selector to ensure consistent feature selection during prediction
joblib.dump(selector, "backend/ml/models/text_feature_selector.pkl")

print(f"✅ Text features converted & saved! Shape: {text_features_selected.shape}")
print(f"✅ Feature selector saved for consistent transformations")