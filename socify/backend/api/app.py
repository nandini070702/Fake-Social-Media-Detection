from flask import Flask
from flask_cors import CORS
from .routes import api_bp  # Import the blueprint from routes

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for allowing frontend requests (e.g., from 127.0.0.1:5500)
CORS(app)

# Register blueprint for /api routes
app.register_blueprint(api_bp, url_prefix="/api")

# Start the Flask app
if __name__ == "__main__":
    print("🚀 Starting Flask Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

