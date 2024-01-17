import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn. naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('spam.csv')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
df.drop_duplicates(inplace=True)

# Display the head of the DataFrame in Streamlit
# st.title('Spam Detection App')
# st.write('Showing the head of the DataFrame:')
# st.dataframe(df.head())

# Our dataset has values labeled as ham or spam, but we will change the label to 0 and 1 (numeric values)
print(df.shape)
df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})
print(df.head())

def clean_data(message):
    message_without_punc= [character for character in message if character not in string.punctuation]
    message_without_punc=''.join(message_without_punc)

    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

#count vectorizer is a method to convert text to numerical data

cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 , random_state =0)

model = MultinomialNB().fit(x_train,y_train)
predictions = model.predict(x_test)


# precision ,recall , f1-score and support
# print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test , predictions))

def predict(text):
    labels =['not spam','spam']
    x = cv.transform([text]).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is : '+labels[v])

# print(predict(["Thankyou for choosing our service"]))


#designing of webapp
st.title('spam classifier')
st.image('spam.jpeg')
st.write("""This Streamlit web app implements a spam classifier using a Multinomial Naive Bayes algorithm. The dataset, containing labeled messages, is preprocessed by removing duplicates and encoding labels. The text data is cleaned, converted into numerical format using CountVectorizer, and split into training and testing sets. A Naive Bayes model is trained on the training set and used to predict whether a user-input message is spam or not. The app provides an interactive interface where users can input a message, click 'predict', and receive the classification result, offering a user-friendly tool for spam detection.""")
user_input = st.text_input('write your message')
submit = st.button('predict')

if submit:
    answer = predict(user_input)
    st.text(answer)

st.write("""by RISHABH GUPTA""")
