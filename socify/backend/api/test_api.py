import requests
import json
import numpy as np

# API URL
url = "http://127.0.0.1:5000/api/predict"

# Different test cases for API testing
test_cases = [
    {
        "name": "✅ Normal Random Data",
        "text_features": np.random.rand(867).tolist(),
        "image_features": np.random.rand(1000).tolist(),
        "profile_url": None  # No URL, only feature-based detection
    },
    {
        "name": "🛑 All Zeros (Edge Case)",
        "text_features": [0] * 867,
        "image_features": [0] * 1000,
        "profile_url": None
    },
    {
        "name": "🛑 All Ones (Extreme Case)",
        "text_features": [1] * 867,
        "image_features": [1] * 1000,
        "profile_url": None
    },
    {
        "name": "🛑 Random Large Values (Model Sensitivity)",
        "text_features": (np.random.rand(867) * 100).tolist(),
        "image_features": (np.random.rand(1000) * 100).tolist(),
        "profile_url": None
    },
    {
        "name": "🛑 Small Values Close to Zero",
        "text_features": (np.random.rand(867) * 0.0001).tolist(),
        "image_features": (np.random.rand(1000) * 0.0001).tolist(),
        "profile_url": None
    },
    {
        "name": "✅ Instagram Profile URL",
        "profile_url": "https://www.instagram.com/__mahiyadav",
        "text_features": None,
        "image_features": None
    },
    {
        "name": "✅ Twitter Profile URL",
        "profile_url": "https://twitter.com/elonmusk",
        "text_features": None,
        "image_features": None
    },
    {
        "name": "✅ Facebook Profile URL",
        "profile_url": "https://www.facebook.com/zuck",
        "text_features": None,
        "image_features": None
    },
    {
        "name": "✅ Mixed Input (URL + Image Features)",
        "profile_url": "https://www.instagram.com/__mahiyadav",
        "text_features": None,
        "image_features": np.random.rand(1000).tolist()
    }
]

headers = {"Content-Type": "application/json"}

# Loop through each test case and send a request
for test in test_cases:
    data = {}

    # Include profile URL if available
    if test["profile_url"]:
        data["profile_url"] = test["profile_url"]

    # Include features only if profile URL is not given
    if test["text_features"] is not None:
        data["text_features"] = test["text_features"]
    if test["image_features"] is not None:
        data["image_features"] = test["image_features"]

    response = requests.post(url, json=data, headers=headers)

    print("\n------------------------------------")
    print(f"Running Test: {test['name']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("------------------------------------\n")
