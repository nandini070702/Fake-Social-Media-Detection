import os
import numpy as np
import joblib

# Adjusted path to work from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "text_model.pkl")

# Example features
sample_text_features = np.array([0.2, 0.5, 0.1, 0.3, 0.0, 1.0]).reshape(1, -1)

# Load and predict
text_model = joblib.load(model_path)
prediction = text_model.predict(sample_text_features)
label = "Real" if prediction[0] == 1 else "Fake"

print("Prediction:", label)
