import joblib

model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

print("✅ Model & Vectorizer loaded successfully!")
