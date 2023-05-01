import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

ps = PorterStemmer()

st.sidebar.image("spam.webp", width=300 )

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Email Spam Classification</h1>", unsafe_allow_html=True)


input_sms = st.text_area("Enter the email", height=100)

if st.button('Verification'):


    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. calculate probability
    try:
        proba = model.predict_proba(vector_input)[0]
    except IndexError:
        proba = None
    # 5. Display
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")
    if proba is not None:
        st.write(f"Probability: {proba[1] * 100:.2f}%")

        fig, ax = plt.subplots()

        bars = ['Spam', 'Not Spam']
        values = [proba[1] * 100, proba[0] * 100]

        ax.bar(bars, values, color=['red', 'green'])
        ax.set_title('Prediction Result')
        ax.set_ylabel('Probability (%)')

        for i, v in enumerate(values):
            ax.text(i, v + 1, str(round(v, 2)), ha='center')

        st.pyplot(fig)

    else:
        st.write("Unable to calculate probability.")
