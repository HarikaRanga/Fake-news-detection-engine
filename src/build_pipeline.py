# build_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
DATA_PATH = r"C:\Users\HP\Desktop\fake_news_detection\data\train.csv"
MODEL_PATH = r"C:\Users\HP\Desktop\fake_news_detection\models\model_pipeline.pkl"

# Create models folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load merged dataset
df = pd.read_csv(DATA_PATH)

# Check label column
print("Labels in dataset:", df['label'].value_counts())

# Balance dataset — equal number of fake and real
min_count = df['label'].value_counts().min()
df_fake = df[df['label'] == "FAKE"].sample(min_count, random_state=42)
df_real = df[df['label'] == "REAL"].sample(min_count, random_state=42)
df_balanced = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)  # shuffle

X = df_balanced['text']
y = df_balanced['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=200))
])

print("Training model...")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)
print("✅ Model retrained successfully and saved to:", MODEL_PATH)

# Optional: test one sample
sample_text = "Health Ministry reports seasonal influenza vaccination coverage exceeded 90% among eligible groups this quarter, reducing hospital admissions."
pred = model.predict([sample_text])[0]
conf = max(model.predict_proba([sample_text])[0]) * 100
print(f"Sample Test Prediction: {pred} ({conf:.2f}% confidence)")
