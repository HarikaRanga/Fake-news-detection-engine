# ==========================================================
# 📰 collect_data.py (Fixed Version)
# ==========================================================

import pandas as pd
import os
import re

# ----------------------------------------------------------
# 📂 Absolute paths
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "clean_news.csv")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ----------------------------------------------------------
# 🧹 Basic text cleaning
# ----------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # remove numbers/special chars
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    return text

# ----------------------------------------------------------
# 📥 Load and merge datasets
# ----------------------------------------------------------
def load_datasets():
    dfs = []

    # Assign labels based on CSV source: 0=fake, 1=real
    sources = [
        ("Fake.csv", 0),
        ("True.csv", 1)
    ]

    for filename, label in sources:
        file = os.path.join(DATA_DIR, filename)
        if os.path.exists(file):
            print(f"✅ Found: {file}")
            df = pd.read_csv(file)

            # Ensure text column exists
            if 'text' not in df.columns:
                if 'title' in df.columns:
                    df.rename(columns={'title': 'text'}, inplace=True)
                else:
                    raise ValueError(f"No 'text' or 'title' column in {filename}")

            df['label'] = label
            dfs.append(df)
        else:
            print(f"⚠️ Warning: File not found - {file}")

    if not dfs:
        raise FileNotFoundError(f"❌ No dataset files found in {DATA_DIR}")

    data = pd.concat(dfs, ignore_index=True)
    return data

# ----------------------------------------------------------
# 🧪 Process and clean
# ----------------------------------------------------------
def preprocess():
    df = load_datasets()
    print(f"✅ Loaded {len(df)} records.")

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Drop empty texts
    df = df[df['text'].str.strip() != ""]

    # Remove duplicates
    df.drop_duplicates(subset=['text'], inplace=True)

    # Balance dataset
    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]
    min_len = min(len(fake_df), len(real_df))
    df_balanced = pd.concat([fake_df.sample(min_len, random_state=42),
                             real_df.sample(min_len, random_state=42)])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save cleaned CSV
    df_balanced.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved cleaned dataset → {OUTPUT_FILE}")
    print("Label distribution:")
    print(df_balanced['label'].value_counts())

if __name__ == "__main__":
    preprocess()
