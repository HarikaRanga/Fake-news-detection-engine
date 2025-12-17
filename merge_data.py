import pandas as pd

# Use the absolute path to your data folder
df_fake = pd.read_csv(r"C:\Users\HP\Desktop\fake_news_detection\data\Fake.csv")
df_true = pd.read_csv(r"C:\Users\HP\Desktop\fake_news_detection\data\True.csv")
# Rename your text columns to 'text' (replace 'headline' if needed)
df_fake = df_fake.rename(columns={"headline":"text"})
df_true = df_true.rename(columns={"headline":"text"})

# Assign labels
df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

# Merge datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Ensure all text is string
df["text"] = df["text"].astype(str)

# Save as train.csv in the same data folder
df[["text","label"]].to_csv(r"C:\Users\HP\Desktop\fake_news_detection\data\train.csv", index=False)
print("Wrote train.csv with", len(df), "rows")
