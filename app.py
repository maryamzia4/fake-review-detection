# app.py

import streamlit as st
import joblib
import nltk
nltk.download('stopwords')
from utils import text_process


# Load the model
model = joblib.load('fake_review_detector_svm.pkl')

st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️‍♂️ Fake Review Detector")
st.markdown("Enter a product review and let the model predict if it's **genuine** or **fake**.")

# Input box
review_input = st.text_area("Enter your review:")

# Predict button
if st.button("Detect"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        result = model.predict([review_input])[0]
        if result == 1:
            st.success("✅ This review seems **Genuine**.")
        else:
            st.error("⚠️ This review is likely **Fake**.")
