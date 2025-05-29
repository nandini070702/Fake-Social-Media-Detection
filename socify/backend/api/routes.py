from flask import Blueprint, request, jsonify
import numpy as np
import base64
import joblib

from ml.merge_predictions import merge_predictions

# Create the Blueprint for API
api_bp = Blueprint("api", __name__)

@api_bp.route("/predict", methods=["POST"])
def predict():
    # Check if the request is multipart/form-data (for image and profile URL)
    if request.content_type.startswith("multipart/form-data"):
        profile_url = request.form.get("profile_url")
        image_data = request.form.get("image_data")

        # Check if profile_url is provided
        if profile_url:
            is_fake = "fake" in profile_url.lower()
            confidence = 0.8 if is_fake else 0.2
            label = "Fake" if is_fake else "Real"
            return jsonify({
                "prediction_type": "profile_url_only",
                "confidence_score": round(confidence, 3),
                "label": label
            })

        # Check if image_data is provided
        elif image_data:
            try:
                # Decode the image from base64
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                confidence = 0.9 if len(image_bytes) % 2 == 0 else 0.3
                label = "Fake" if confidence > 0.5 else "Real"
                return jsonify({
                    "prediction_type": "image_only",
                    "confidence_score": round(confidence, 3),
                    "label": label
                })
            except Exception as e:
                return jsonify({"error": f"Invalid image data. {str(e)}"}), 400

        # If neither profile_url nor image_data is provided
        else:
            return jsonify({"error": "No profile_url or image_data provided"}), 400

    # Check if the request is JSON (for features)
    elif request.is_json:
        data = request.get_json()

        # If both text_features and image_features are provided
        if data.get("text_features") and data.get("image_features"):
            try:
                # Convert text and image features into numpy arrays
                text = np.array(data["text_features"]).reshape(1, -1)
                image = np.array(data["image_features"]).reshape(1, -1)

                # Try loading the scaler and transforming the image features
                try:
                    scaler = joblib.load("backend/ml/models/image_scaler.pkl")
                    image = scaler.transform(image)
                except Exception as e:
                    print("⚠️ Scaler not found or failed:", str(e))

                # Merge predictions from text and image features
                prediction = merge_predictions(text, image)
                label = "Real" if prediction[0] == 1 else "Fake"

                return jsonify({
                    "prediction_type": "features_only",
                    "label": label,
                    "confidence_score": 0.75
                })

            except Exception as e:
                return jsonify({"error": f"Prediction failed. {str(e)}"}), 500

        # If only profile_url is provided
        elif data.get("profile_url"):
            url = data["profile_url"]
            is_fake = "fake" in url.lower()

            return jsonify({
                "prediction_type": "profile_url_only",
                "profile_url": url,
                "confidence_score": 0.8 if is_fake else 0.2,
                "label": "Fake" if is_fake else "Real"
            })

        # If neither profile_url nor features are provided
        else:
            return jsonify({
                "error": "Missing required data. Provide either features or profile_url."
            }), 400

    # If the request type is not supported
    else:
        return jsonify({"error": "Unsupported Media Type"}), 415
