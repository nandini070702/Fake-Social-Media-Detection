import joblib
import os

# Dynamically get the correct path to the models directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/api/
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../ml/models"))  # backend/ml/models

# Model names
models = {
    "text_xgb": "xgboost_text.pkl",
    "text_rf": "random_forest_text.pkl",
    "image_xgb": "xgboost_image.pkl",
    "image_rf": "random_forest_image.pkl",
    "text_selector": "feature_selector.pkl"
}

# Load models with better error handling
loaded_models = {}

for model_name, file_name in models.items():
    model_path = os.path.join(MODEL_DIR, file_name)
    try:
        loaded_models[model_name] = joblib.load(model_path)
        print(f"✅ Successfully loaded {file_name}")
    except FileNotFoundError:
        print(f"❌ Error: {file_name} not found in {MODEL_DIR}")
        loaded_models[model_name] = None
    except Exception as e:
        print(f"⚠️ Error loading {file_name}: {e}")
        loaded_models[model_name] = None

# Assign loaded models to variables
text_xgb = loaded_models["text_xgb"]
text_rf = loaded_models["text_rf"]
image_xgb = loaded_models["image_xgb"]
image_rf = loaded_models["image_rf"]
text_selector = loaded_models["text_selector"]

