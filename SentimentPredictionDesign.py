import streamlit as st
import joblib
import numpy as np

# Load the CountVectorizer and Logistic Regression model during deployment
count_vectorizer = joblib.load("count_vectorizer.pkl")
lr_model = joblib.load("Movies_Review_Regression.pkl")

# Set a custom page title and icon
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon=":clapper:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar to allow users to choose color palette
st.sidebar.title("Customize Your App")
bg_color = st.sidebar.color_picker("Choose Background Color", "#f5f5f5")
text_color = st.sidebar.color_picker("Choose Text Color", "#333333")

# Dummy variable to trigger script rerun
rerun_trigger = st.sidebar.button("Rerun", key="rerun_button")

# Set the custom color palette
st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            color: {text_color};
        }}
        .stButton {{
            background-color: inherit;
            color: inherit;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add an emoji icon and a clear heading
st.title(":clapper: Movie Based Sentiments Reviews")
st.header("Checking the Sentiments of the Reviews")

def test_model(sentence):
    sen = count_vectorizer.transform([sentence]).toarray()
    res = lr_model.predict(sen)[0]
    return 'Positive review' if res == 1 else 'Negative review'

def main():
    # Use responsive design with columns
    col1, col2 = st.columns(2)
    user_input = col1.text_area("Enter your movie review:")

    # Use a button with a custom color
    if col2.button("Predict", key="predict_button"):
        if user_input:
            prediction = test_model(user_input)
            # Display emojis based on sentiment
            emoji = "😃" if prediction == "Positive review" else "😢"
            st.success(f"The sentiment of the review is: {prediction} {emoji}")
        else:
            st.warning("Please enter a movie review.")

if __name__ == "__main__":
    main()
