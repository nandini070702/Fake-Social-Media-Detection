import numpy as np
import joblib

# Example image features (replace with actual test features from your dataset)
sample_image_features = np.array([
    0.6, 0.7, 0.1, 0.4, 0.9, 0.3  # Dummy features
]).reshape(1, -1)

# Load scaler and model
scaler = joblib.load("ml/models/image_scaler.pkl")
model = joblib.load("ml/models/image_model.pkl")

# Scale and predict
scaled_features = scaler.transform(sample_image_features)
prediction = model.predict(scaled_features)
label = "Real" if prediction[0] == 1 else "Fake"

print("Prediction:", label)
