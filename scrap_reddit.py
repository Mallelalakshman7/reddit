import praw
import pandas as pd

reddit=praw.Reddit(client_id='w-H3wffb9s6nrOzq8p04Tw',client_secret='99vqay0FggpVckjRM1LU1SmAOK10Lg',user_agent='my_reddit_app v1.0 by /u/EastButterscotch3819')

def scarpe_reddit(subreddit_name,post_count):
    subreddit=reddit.subreddit(subreddit_name)
    posts=[]
    for post in subreddit.hot(limit=post_count):
        posts.append(post.title + " "+post.selftext)
    return pd.DataFrame({'text':posts})

if __name__=="__main__":
    subreddit_name='python'
    post_count=100
    reddit_data=scrape_reddit(subreddit_name,post_count)
    print(reddit_data.head())
    

