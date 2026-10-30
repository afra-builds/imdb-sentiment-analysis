# IMDb Movie Sentiment Analysis

## Project Overview

This project is a machine learning-based sentiment analysis system that predicts whether a movie review is positive or negative.
It uses Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier to analyze movie reviews.
The project also includes a simple interactive web application built with Streamlit.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes
- Streamlit
- Joblib
- Git and GitHub
## Dataset

- Dataset Name: IMDb Movie Reviews Dataset
- Total Reviews: 50,000
- Duplicate Records Removed: 418
- Cleaned Records: 49,582
- Features: Review and Sentiment
- Sentiments: Positive and Negative

The dataset contains movie reviews labeled as positive or negative. It was cleaned by removing duplicate records before model training.
## Model Training

- Train-Test Split: 80% Training and 20% Testing
- Training Samples: 39,665
- Testing Samples: 9,917
- Text Vectorization: CountVectorizer
- Classifier: Multinomial Naive Bayes

## Model Evaluation

- Accuracy: 85.19%

The model was evaluated using unseen test data. A confusion matrix was also used to understand correct and incorrect predictions.
## How to Run the Project

1. Clone the repository:

   git clone https://github.com/afra-builds/imdb-sentiment-analysis.git

2. Open the project folder.

3. Install the required libraries:

   pip install pandas scikit-learn streamlit joblib

4. Run the Streamlit application:

   python -m streamlit run app.py

5. Open the local URL in your browser.
## Features

- Predicts positive and negative movie reviews.
- Accepts user input through a text area.
- Displays clear sentiment predictions.
- Handles empty review input.
- Uses a saved machine learning model.
- Provides an interactive Streamlit interface.

## Limitations

- The model may not correctly classify ambiguous reviews.
- Sarcasm and complex language can affect predictions.
- The model is trained on IMDb movie reviews, so it may not perform equally well on other types of customer feedback.
## Sample Predictions

| Movie Review | Predicted Sentiment |
|---|---|
| This movie was amazing and I loved the story. | Positive |
| This movie was boring and terrible. | Negative |
| The movie was okay, but some scenes were good and others were boring. | Negative |

## Dataset Source

The dataset was downloaded from Kaggle:

[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset contains 50,000 movie reviews labeled as positive or negative.