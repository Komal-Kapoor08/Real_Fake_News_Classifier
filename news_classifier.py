import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding



df = pd.read_csv('Fake_Real_Data.csv')
df.drop_duplicates(inplace = True)
df['Label'] = df['label'].map({'Fake':0,'Real':1})

st.title('Fake news detection')
st.write('This application is working on LLMS so might be it takes Time')
user = st.text_input('enter news')


t = Tokenizer(num_words = 5000)
t.fit_on_texts(df['Text'])

seq = t.texts_to_sequences(df['Text'])
x = pad_sequences(seq, maxlen= 200)
y = np.array(df['Label'])

model = Sequential([
    Embedding(input_dim= 5000, output_dim = 64, input_length = 200),
    LSTM(32, return_sequences = True),
    LSTM(32),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

model.fit(x,y, epochs = 1)

def classify(news):
    seq = t.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen= 200)
    pred = model.predict(padded)[0][0]
    return 'Real' if pred>0.5 else 'Fake'


if user:
    result = classify(user)
    st.write(f'This news is {result}')

hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
