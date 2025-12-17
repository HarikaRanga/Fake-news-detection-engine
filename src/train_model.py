# ==========================================================
# 🧠 train_model.py (Enhanced Version)
# Train Fake News Detection Model (TF-IDF + Logistic Regression)
# Author: Ranga Harika
# ==========================================================

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 📂 Absolute paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "clean_news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABELS_FILE = os.path.join(MODEL_DIR, "label_mapping.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 🚀 Train Model
# ---------------------------------------------------------
def train():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Dataset not found! Expected at: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    X = df['text']
    y = df['label']  # 0 = FAKE, 1 = REAL

    print(f"🧩 Loaded {len(df)} samples for training...")
    print("Numeric label values found:", sorted(y.unique()))
    
    # Save label mapping for consistent prediction
    label_mapping = {0: 'FAKE', 1: 'REAL'}
    joblib.dump(label_mapping, LABELS_FILE)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=30000,   # increased features
        min_df=5,             # ignore very rare words
        ngram_range=(1,2),    # unigrams + bigrams
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression Model
    model = LogisticRegression(
        max_iter=500,
        class_weight='balanced',  # handles class imbalance
        solver='liblinear'        # works well for small/medium datasets
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'],
                yticklabels=['FAKE', 'REAL'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Show some sample predictions
    sample_df = pd.DataFrame({'text': X_test[:5], 'true_label': y_test[:5], 'pred_label': y_pred[:5]})
    print("\nSample test predictions (true -> pred):\n", sample_df)

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"💾 Model saved to: {MODEL_FILE}")
    print(f"💾 Vectorizer saved to: {VECTORIZER_FILE}")
    print(f"💾 Label mapping saved to: {LABELS_FILE}")

if __name__ == "__main__":
    train()


