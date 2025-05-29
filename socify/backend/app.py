from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/socialverify"
mongo = PyMongo(app)

# ----------------------------
# AUTH ROUTES
# ----------------------------

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    email = data['email']
    password = data['password']

    if mongo.db.users.find_one({'email': email}):
        return jsonify({'error': 'User already exists!'}), 400

    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one({
        'username': username,
        'email': email,
        'password': hashed_password
    })

    return jsonify({'message': 'User registered successfully!'}), 201

@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    username = data['username']
    password = data['password']

    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        return jsonify({'message': 'Login successful!'}), 200
    else:
        return jsonify({'error': 'Invalid username or password'}), 401

# ----------------------------
# FAKE PROFILE DETECTION ROUTE
# ----------------------------

@app.route('/detect', methods=['POST'])
def detect_profile():
    data = request.get_json()
    url = data.get('url', '')

    # Dummy logic for detection (replace with actual ML code later)
    if 'fake' in url.lower():
        prediction = 'Fake Profile'
        confidence = 0.1
    else:
        prediction = 'Real Profile'
        confidence = 0.9

    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)

