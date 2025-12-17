# preprocess.py
import pandas as pd
import os

def preprocess_data(fake_path, real_path, save_path):
    # Load datasets
    print("Loading datasets...")
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Add labels: 0 = FAKE, 1 = REAL
    fake_df['label'] = 0
    real_df['label'] = 1

    # Keep only title + text combined
    fake_df['text'] = fake_df['title'] + " " + fake_df['text']
    real_df['text'] = real_df['title'] + " " + real_df['text']

    # Combine and shuffle
    df = pd.concat([fake_df[['text', 'label']], real_df[['text', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save preprocessed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Preprocessing complete! File saved to: {save_path}")

if __name__ == "__main__":
    fake_path = r"C:\Users\HP\Desktop\fake_news_detection\data\raw\Fake.csv"
    real_path = r"C:\Users\HP\Desktop\fake_news_detection\data\raw\True.csv"
    save_path = r"C:\Users\HP\Desktop\fake_news_detection\data\processed\train.csv"

    preprocess_data(fake_path, real_path, save_path)
