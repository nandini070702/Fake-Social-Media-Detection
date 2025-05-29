import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from merge_predictions import merge_predictions
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
from sklearn.preprocessing import StandardScaler

# Load processed datasets
text_features = np.load("datasets/text_dataset/processed_bot_data.npy")
image_features = np.load("datasets/image_dataset/processed_faces/image_features.npy")
true_labels = np.load("datasets/image_dataset/processed_faces/image_labels.npy")

# ✅ Limit dataset sizes to match
num_samples = min(len(text_features), len(image_features), len(true_labels))
text_features = text_features[:num_samples]
image_features = image_features[:num_samples]
true_labels = true_labels[:num_samples]

# Check class distribution
unique, counts = np.unique(true_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"✅ Class distribution: {class_distribution}")

# Create and save feature selector if not already done
try:
    text_selector = joblib.load("backend/ml/models/text_feature_selector.pkl")
    print("✅ Loaded existing feature selector")
    text_features_selected = text_selector.transform(text_features)
except:
    print("⚠️ Creating new feature selector")
    text_selector = SelectKBest(mutual_info_classif, k=500)
    text_features_selected = text_selector.fit_transform(text_features, true_labels)
    joblib.dump(text_selector, "backend/ml/models/text_feature_selector.pkl")

# Optional: Create and save scalers for future use
text_scaler = StandardScaler()
image_scaler = StandardScaler()
text_features_scaled = text_scaler.fit_transform(text_features_selected)
image_features_scaled = image_scaler.fit_transform(image_features)
joblib.dump(text_scaler, "backend/ml/models/text_scaler.pkl")
joblib.dump(image_scaler, "backend/ml/models/image_scaler.pkl")

# Get final predictions
final_predictions = []

# Process in batches to improve efficiency
batch_size = 100
for i in range(0, num_samples, batch_size):
    end_idx = min(i + batch_size, num_samples)
    text_batch = text_features[i:end_idx]
    image_batch = image_features[i:end_idx]
    batch_predictions = merge_predictions(text_batch, image_batch)
    final_predictions.extend(batch_predictions)

# Convert to NumPy array
final_predictions = np.array(final_predictions)

# Evaluate Accuracy
accuracy = accuracy_score(true_labels, final_predictions)
print(f"✅ Merged Model Accuracy: {accuracy:.4f}")

# Detailed evaluation
conf_matrix = confusion_matrix(true_labels, final_predictions)
print("\n✅ Confusion Matrix:")
print(conf_matrix)
print("\n✅ Classification Report:")
print(classification_report(true_labels, final_predictions, zero_division=0))

# Calculate accuracy for each class
class_0_accuracy = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
class_1_accuracy = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
print(f"✅ Class 0 (Fake) Accuracy: {class_0_accuracy:.4f}")
print(f"✅ Class 1 (Real) Accuracy: {class_1_accuracy:.4f}")