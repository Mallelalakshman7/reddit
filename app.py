import streamlit as st
import praw
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from app1 import preprocess_text, clean_text, extract_features, classifier


# Sidebar for subreddit selection
subreddit_name = st.sidebar.text_input("Enter Subreddit Name", "Knitting")
num_posts = st.sidebar.slider("Number of Posts", 100, 1000, 500)

# Main section
st.title("Reddit Sentiment Analysis")
st.write(f"Analyzing {num_posts} posts from r/{subreddit_name}")

# Reddit API setup
reddit = praw.Reddit(client_id='w-H3wffb9s6nrOzq8p04Tw', client_secret='99vqay0FggpVckjRM1LU1SmAOK10Lg', user_agent='my_reddit_app v1.0 by /u/EastButterscotch3819')
subreddit = reddit.subreddit(subreddit_name)
posts = subreddit.hot(limit=num_posts)

# Data processing
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
df1['created_date'] = pd.to_datetime(df1['created'], unit='s')
df1['cleaned_text'] = df1['body'].apply(clean_text)

# Display DataFrame
st.write(df1.head())

# Sentiment Analysis
predictions = [classifier.classify(extract_features(tokenize_words(text))) for text in df1['cleaned_text']]
df1['predicted_sentiment'] = predictions

# Display Sentiment Distribution
st.bar_chart(df1['predicted_sentiment'].value_counts())

# Word Cloud
all_text = ' '.join([text for text in df1['cleaned_text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Save results
if st.button('Save Results to CSV'):
    df1.to_csv('predicted_sentiments.csv', index=False)
    st.write("Results saved successfully!")
