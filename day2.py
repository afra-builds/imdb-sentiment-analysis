
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv("IMDB Dataset.csv")
df = df.dropna()
df = df.drop_duplicates()

# Separate reviews and labels
X = df["review"]
y = df["sentiment"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
print("Data split completed!")

from sklearn.feature_extraction.text import CountVectorizer

# Convert text into numbers
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training data shape:", X_train_vec.shape)
print("Testing data shape:", X_test_vec.shape)
print("Text converted into numbers successfully!")

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create the model
model = MultinomialNB()

# Train the model
model.fit(X_train_vec, y_train)

# Predict sentiments for test reviews
y_pred = model.predict(X_test_vec)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model training completed!")
print("Model Accuracy:", accuracy)
print("Model Accuracy Percentage:", round(accuracy * 100, 2), "%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

print("Confusion Matrix:")
print(cm)
print("Labels order: negative, positive")

# Find incorrect predictions
results = pd.DataFrame({
    "Review": X_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

wrong_predictions = results[
    results["Actual"] != results["Predicted"]
]

print("\nTotal wrong predictions:", len(wrong_predictions))
print("\nFirst 5 wrong predictions:")
print(wrong_predictions.head(5).to_string(index=False))

print("\nAnalysis of 5 wrong predictions:")

for _, row in wrong_predictions.head(5).iterrows():
    print("\nActual:", row["Actual"])
    print("Predicted:", row["Predicted"])
    print("Review:", row["Review"][:250], "...")

def predict_sentiment(review):
    if not review or not review.strip():
        return "Please enter a review!"

    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    return prediction


# Test the function
print("\nTesting prediction function:")

print(predict_sentiment(
    "This movie was amazing and I loved it!"
))

print(predict_sentiment(
    "This movie was boring and terrible."
))

print(predict_sentiment(""))


