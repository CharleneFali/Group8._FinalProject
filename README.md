# Group8._FinalProject
Movie Sentiment Analysis
This project involves sentiment analysis for movie reviews using various machine learning models and a deep learning approach. The code includes data preprocessing, exploratory data analysis (EDA), model training, and deployment of a Streamlit app for real-time sentiment prediction.
Table of Contents
1.	Streamlit App
2.	Machine Learning Models
3.	Deep Learning Model
4.	Data Preprocessing
5.	Model Training
6.	Model Deployment
7.	Evaluation
8.	Usage Example
9.	Dependencies
10.	How to Run
11.	Author
12.	Link to Explanatory video
Streamlit App
The streamlit_app.py file contains the code for a Streamlit web application. Users can input movie reviews, and the sentiment analysis model predicts whether the review is positive or negative.
Machine Learning Models
Machine learning models such as Multinomial Naive Bayes, Random Forest, and Logistic Regression are implemented using scikit-learn. Hyperparameter tuning is performed to enhance model accuracy.
Deep Learning Model
A deep learning model using LSTM layers is implemented with TensorFlow and Keras for more complex sentiment analysis.
Data Preprocessing
Text data is preprocessed by removing stop words, converting text to lowercase, and tokenizing words using NLTK. The dataset is balanced by combining positive and negative samples.
Model Training
The machine learning models are trained on the IMDb dataset. Grid search is utilized for hyperparameter tuning.
Model Deployment
The best Logistic Regression model is saved along with the CountVectorizer for deployment. A deep learning model is saved as an H5 file.
Evaluation
The performance of each model is evaluated using accuracy scores and classification reports.
Usage Example
A simple example code demonstrates how to load the models during deployment and make predictions on new data.
Dependencies
Ensure the following Python libraries are installed:
•	pandas
•	numpy
•	scikit-learn
•	nltk
•	matplotlib
•	seaborn
•	streamlit
•	tensorflow
Install the dependencies using:
bashCopy code
pip install -r requirements.txt 
How to Run
Mount Google Drive using the provided Colab code.
Execute the Streamlit app: streamlit run streamlit_app.py
Enter a movie review in the app, click "Predict," and view the sentiment prediction.
Authors
[Glenn Bartels Odoom]
[Charlene Esi Duametu]
Below is url the YouTube video explaining how our application works
https://youtu.be/6YHm1IzN02E

