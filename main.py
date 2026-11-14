
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# Load the IMDB dataset word index
word_index = imdb.get_word_index()

reverse_word_index = {
    value: key for (key, value) in word_index.items()
}


# Load the pre-trained SimpleRNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'simple_rnn_imdb_compatible.h5',
        compile=False
    )


model = load_model()


# Function to decode the review from integers to words
def decode_review(encoded_review):

    return ' '.join([
        reverse_word_index.get(i - 3, '?')
        if i >= 4 else '?'
        for i in encoded_review
    ])


# Function to preprocess user input
def preprocess_text(text):

    # Tokenize the text into words
    words = text_to_word_sequence(text)

    # Convert words to integers using the IMDB word index
    # 2 represents unknown words
    encoded_review = [
        word_index[word] + 3 if word in word_index else 2
        for word in words
    ]

    # Pad the sequence to a fixed length of 500
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=500,
        padding='pre',
        truncating='pre'
    )

    return padded_review


# Prediction Function
def predict_sentiment(review):

    # Preprocess the review
    processed_review = preprocess_text(review)

    # Make prediction
    prediction = model.predict(processed_review, verbose=0)

    # Interpret the prediction
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, float(prediction[0][0])


# Streamlit Application
st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to predict its sentiment "
    "(Positive or Negative)."
)


# User input
user_input = st.text_area("Movie Review", "")


# Predict Sentiment
if st.button("Predict Sentiment"):

    if not user_input.strip():

        st.warning("Please enter a movie review.")

    else:

        sentiment, score = predict_sentiment(user_input)

        st.subheader("Prediction Result")

        if sentiment == "Positive":

            st.success("Sentiment: Positive")
            confidence = score

        else:

            st.error("Sentiment: Negative")
            confidence = 1 - score

        st.write(f"Prediction Score: {score:.4f}")

        st.write(f"Confidence: {confidence * 100:.2f}%")
