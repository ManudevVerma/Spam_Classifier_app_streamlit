import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("punkt_tab")
nltk.download("stopwords")

tfidf = pickle.load(open("vectorize.pkl", 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Classifier')

input_sms = st.text_area('Enter the message')

# 1. Preprocess

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    ps = PorterStemmer()
    for i in text:
        if (i.isalnum()) and (i not in stopwords.words('english')) and (i not in string.punctuation):
            i = ps.stem(i)
            y.append(i)

    return " ".join(y)

transformed_sms = transform_text(input_sms)
# 2. Vectorize
vector_input = tfidf.transform([transformed_sms])
# 3. Predict
result = model.predict(vector_input)[0]
# 4. Display
if st.button('Predict'):
    if not input_sms:
        st.header("Input a message.")
    elif result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')



