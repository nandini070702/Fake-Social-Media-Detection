from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB URI (change if using MongoDB Atlas)
app.config["MONGO_URI"] = "mongodb://localhost:27017/socialverify"
mongo = PyMongo(app)

# Sign Up API
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

# Sign In API
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

if __name__ == '__main__':
    app.run(debug=True)
