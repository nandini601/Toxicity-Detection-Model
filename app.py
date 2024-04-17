import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and TF-IDF vectorizer
model = pickle.load(open("toxicity_model.pkt", "rb"))
tf_idf = pickle.load(open("tf_idf.pkt", "rb"))

# Function to preprocess input text
def prepare_text(text):
    # Your text preprocessing code here
    return text

# Function to predict toxicity
def predict_toxicity(text):
    cleaned_text = prepare_text(text)
    tfidf_text = tf_idf.transform([cleaned_text])
    prediction = model.predict_proba(tfidf_text)[0][1]
    return prediction

# Streamlit app
def main():
    st.title("Toxic Comment Detection")

    # Add background image
    st.image("toxic.png", use_column_width=True)

    # Text input for user to enter comment
    comment = st.text_input("Enter your comment:")

    if st.button("Predict"):
        # Predict toxicity
        prediction = predict_toxicity(comment)
        st.write("Toxicity Score:", prediction)

        # Categorize toxicity level
        if prediction < 0.5:
            st.write("This comment is not toxic.")
        else:
            st.write("This comment is toxic.")

if __name__ == "__main__":
    main()
