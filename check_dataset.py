from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/emi_prediction_dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum()[lambda x: x > 0])

print("\nDuplicate rows:", df.duplicated().sum())

targets = ["emi_eligibility", "max_monthly_emi"]
print("\nTarget column checks:")
for col in targets:
    print(f"{col}: {'FOUND' if col in df.columns else 'MISSING'}")
