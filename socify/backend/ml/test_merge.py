import numpy as np
from merge_predictions import merge_predictions

# Simulated test input (replace with real preprocessed data)
text_sample = np.random.rand(1, 500)  # Simulating text features
image_sample = np.random.rand(1, 1000)  # Simulating image features from ResNet

# Get merged predictions
final_prediction = merge_predictions(text_sample, image_sample)

print("🔹 Final Profile Classification:", "REAL" if final_prediction[0] == 1 else "FAKE")
