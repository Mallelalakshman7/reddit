import streamlit as st
from scrap_reddit import scrap_reddit
from analyze_sentiment import analyze_reddit_data
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title('Reddit Sentiment Analysis')

subreddit_name=st.sidebar.text_input('Subreddit','python')
post_count=st.sidebar.slider('Number of Posts',10,100,50)

if st.button('Scrape and Analyze'):
    with st.spinner('Scrapiing Reddit...'):
        reddit_data=scrap_reddit(subreddit_name,post_count)
        reddit_data=analyze_reddit_data(reddit_data)

    st.success('Scraping and Analysis Complete!')


    st.subheader('Sentiment Distribution')
    st.bar_chart(reddit_data['sentiment'].value_counts())

    st.subheader('WordCloud for Positive Sentiment')
    positive_text=' '.join(reddit_data[reddit_data['sentiment']=='1']['text'])
    wordcloud=WordCloud(width=800,height=400,background_color='white').generate(positive_text)
    st.image(wordcloud.to_array())

st.sidebar.markdown("**Sentiment Labels**:\n 1- Positive \n 0- Negative")

