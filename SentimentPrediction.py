import streamlit as st
import joblib
import numpy as np


count_vectorizer = joblib.load("count_vectorizer.pkl")
lr_model = joblib.load("Movies_Review_Regression.pkl")

st.title("Movie Based Sentiments Reviews")
st.header("Checking the Sentiments of the Reviews")



def test_model(sentence):
    sen = count_vectorizer.transform([sentence]).toarray()
    res = lr_model.predict(sen)[0]
    return 'Positive review' if res == 1 else 'Negative review'

def main():
    st.title("Sentiment Prediction App")

    user_input = st.text_area("Enter your movie review:")
    if st.button("Predict"):
        if user_input:
            prediction = test_model(user_input)
            st.success(f"The sentiment of the review is: {prediction}")
        else:
            st.warning("Please enter a movie review.")

if __name__ == "__main__":
    main()