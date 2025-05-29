from pymongo import MongoClient

MONGO_URI = "mongodb+srv://shrivastavn2002:nandinifakedb@fakeprofiledb.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI)
    db = client["FakeProfileDB"]
    print("✅ Connected to MongoDB. Collections:", db.list_collection_names())
except Exception as e:
    print("❌ MongoDB Connection Failed:", e)
