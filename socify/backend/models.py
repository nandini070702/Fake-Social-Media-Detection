import joblib

def load_models():
    # Load the saved models (make sure the paths are correct)
    random_forest_model = joblib.load('backend/models/rf_model.pkl')  # Update this path
    xgboost_model = joblib.load('backend/models/xgb_model.pkl')  # Update this path
    scaler = joblib.load('backend/models/text_scaler.pkl')  # Update this path
    selector = joblib.load('backend/models/text_selector.pkl')  # Update this path
    
    return random_forest_model, xgboost_model, scaler, selector
