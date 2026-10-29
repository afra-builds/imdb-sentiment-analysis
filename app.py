
import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("🎬 IMDb Movie Sentiment Analysis")

st.write("Enter a movie review to predict its sentiment.")

# User input
review = st.text_area("Write your movie review:")

# Prediction button
if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review!")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == "positive":
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
