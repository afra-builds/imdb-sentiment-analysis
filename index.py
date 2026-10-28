
import pandas as pd

# Load the dataset
df = pd.read_csv("IMDB Dataset.csv")

# Show first 5 reviews
print(df.head())

# Show dataset size
print("Dataset shape:", df.shape)

# Show column names
print("Columns:", df.columns)

# Count positive and negative reviews
print(df["sentiment"].value_counts())

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove missing and duplicate rows
df = df.dropna()
df = df.drop_duplicates()

# Show cleaned dataset size
print("Shape after cleaning:", df.shape)

print("\nSample reviews after cleaning:")
print(df.head())

print("\nSentiment distribution after cleaning:")
print(df["sentiment"].value_counts())

