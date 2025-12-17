import pandas as pd

# Read the separate Excel files
df_fake = pd.read_excel(r".\data\Fake.xlsx")
df_true = pd.read_excel(r".\data\True.xlsx")

# Rename your text columns to 'text' (replace 'headline' if your column name is different)
df_fake = df_fake.rename(columns={"headline":"text"})
df_true = df_true.rename(columns={"headline":"text"})

# Assign labels
df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

# Merge datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Ensure all text is string
df["text"] = df["text"].astype(str)

# Save as train.csv
df[["text","label"]].to_csv(r".\data\train.csv", index=False)
print("Wrote .\\data\\train.csv with", len(df), "rows")
