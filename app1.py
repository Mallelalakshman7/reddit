# -*- coding: utf-8 -*-
"""
Project.py
"""

import os
import zipfile
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import praw
import datetime
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Install packages manually using pip (not in script, should be done outside the script)
# pip install kaggle praw wordcloud matplotlib nltk scikit-learn

# Setup for Kaggle API and download dataset
def setup_kaggle():
    os.makedirs('~/.kaggle', exist_ok=True)
    # Ensure kaggle.json is in the same directory as this script or provide correct path
    os.rename('kaggle.json', '~/.kaggle/kaggle.json')
    os.chmod('~/.kaggle/kaggle.json', 0o600)

def download_dataset():
    import kaggle
    kaggle.api.dataset_download_files('kazanova/sentiment140', path='data', unzip=True)

setup_kaggle()
download_dataset()

# Load and preprocess the dataset
column_names = ['zero', 'ID', 'Date', 'Query', 'Name', 'Message']
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=column_names)

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stop_word = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_word]
    return ' '.join(tokens)

def process_chunk(chunk):
    chunk['preprocess_text'] = chunk['Message'].apply(preprocess_text)
    return chunk

chunk_size = 10000
chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]

with Pool() as pool:
    processed_chunks = pool.map(process_chunk, chunks)

df_processed = pd.concat(processed_chunks)

# Training the sentiment analysis model
training_data = list(zip(df_processed['preprocess_text'], df_processed['zero']))

def tokenize_words(text):
    return word_tokenize(text)

def extract_features(words):
    return {word: True for word in words}

feature_sets = [(extract_features(tokenize_words(text)), label) for text, label in training_data]

train_set, test_set = train_test_split(feature_sets, test_size=0.2, random_state=42)
classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Reddit API setup and data extraction
client_id = 'w-H3wffb9s6nrOzq8p04Tw'
client_secret = '99vqay0FggpVckjRM1LU1SmAOK10Lg'
user_agent = 'my_reddit_app v1.0 by /u/EastButterscotch3819'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
subreddit = reddit.subreddit('Knitting')
posts = subreddit.hot(limit=1000)

data = []
for post in posts:
    post_data = {
        'title': post.title,
        'score': post.score,
        'id': post.id,
        'url': post.url,
        'comms_num': post.num_comments,
        'created': post.created,
        'body': post.selftext
    }
    data.append(post_data)

df1 = pd.DataFrame(data)
df1['created_date'] = df1['created'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df1['month_year'] = df1['created_date'].dt.to_period('M')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

df1['cleaned_text'] = df1['body'].apply(clean_text)

preprocessed_texts = [preprocess_text(text) for text in df1['cleaned_text']]
features = [extract_features(text) for text in preprocessed_texts]
predictions = [classifier.classify(feature) for feature in features]

df1['predicted_sentiment'] = predictions
df1.to_csv('predicted_sentiments.csv', index=False)

# Display results
print(df1[['body', 'predicted_sentiment']])
print(df1.head(50))
print(df1.info())

df1['predicted_sentiment'] = df1['predicted_sentiment'].replace(4, 1)
print(df1['predicted_sentiment'].unique())

# Word cloud visualization
all_text = ' '.join([text for text in df1['cleaned_text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_text)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
