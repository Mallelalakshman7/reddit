import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

stop_word=set(stopwords.words('english'))

with open('sentiment_model.pkl','rb') as model_file:
    classifier=pickle.load(model_file)

def preprocess_text(text):
    text=text.lower()
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word.isalpha() and word not in stop_word]
    return ' '.join(tokens)

def extract_feautres(words):
    return {word: True for word in words}

def predicted_sentiment(text):
    preprocessed_text=preprocess_text(text)
    tokens=word_tokenize(preprocessed_text)
    features=extract_feautres(tokens)
    return classifier.classify(features)

def analyze_reddit_data(reddit_data):
    reddit_data['sentiment']=reddit_data['text'].apply(predicted_sentiment)
    return reddit_data