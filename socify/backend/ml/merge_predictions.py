import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained models
text_xgb = joblib.load("backend/ml/models/xgboost_text.pkl")
text_rf = joblib.load("backend/ml/models/random_forest_text.pkl")
image_xgb = joblib.load("backend/ml/models/xgboost_image.pkl")
image_rf = joblib.load("backend/ml/models/random_forest_image.pkl")

# Load feature selector
try:
    text_selector = joblib.load("backend/ml/models/text_feature_selector.pkl")
    use_saved_selector = True
except:
    use_saved_selector = False
    print("Warning: Feature selector not found, using raw features")

# Define balanced weights
WEIGHTS = {
    "text_xgb": 0.25,
    "text_rf": 0.25,
    "image_xgb": 0.25,
    "image_rf": 0.25
}

def merge_predictions(text_features, image_features):
    """
    Merge text and image model predictions with a balanced approach.
    """
    # Apply feature selection if available
    if use_saved_selector and hasattr(text_selector, 'transform'):
        text_features = text_selector.transform(text_features)
    else:
        # Ensure only the first 500 text features are used
        text_features = text_features[:, :500]
    
    # Get individual model predictions
    text_xgb_pred = text_xgb.predict_proba(text_features)
    text_rf_pred = text_rf.predict_proba(text_features)
    image_xgb_pred = image_xgb.predict_proba(image_features)
    image_rf_pred = image_rf.predict_proba(image_features)
    
    # Create an ensemble using a simple direct voting mechanism
    # Count votes for class 1 (Real)
    votes = np.zeros(len(text_features))
    
    # Get binary predictions from each model with appropriate thresholds
    text_xgb_vote = (text_xgb_pred[:, 1] > 0.5).astype(int)
    text_rf_vote = (text_rf_pred[:, 1] > 0.5).astype(int)
    image_xgb_vote = (image_xgb_pred[:, 1] > 0.5).astype(int)
    image_rf_vote = (image_rf_pred[:, 1] > 0.5).astype(int)
    
    # Sum the votes (1 vote per model)
    votes = text_xgb_vote + text_rf_vote + image_xgb_vote + image_rf_vote
    
    # Predict class 1 (Real) if at least 2 models vote for it
    # This is a simple majority voting system
    final_predictions = (votes >= 2).astype(int)
    
    return final_predictions