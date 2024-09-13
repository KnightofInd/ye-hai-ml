import json
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
import os
from collections import Counter

# Load JSON data
with open('kya yaar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Ensure 'content' is in DataFrame
if 'content' not in df.columns:
    print("Error: 'content' column is missing from the data.")
    exit()

# Create a dedicated folder for plots
output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

# Topic Classification
def classify_topics(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['content'])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    df['topic'] = kmeans.labels_
    
    # Determine topic names based on top 2 keywords
    terms = vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_
    topic_names = {}
    
    for i, center in enumerate(cluster_centers):
        top_indices = center.argsort()[-2:][::-1]  # Top 2 terms
        top_terms = [terms[idx] for idx in top_indices]
        topic_names[i] = " ".join(top_terms)
    
    return df, kmeans, topic_names

# Engagement Metrics
def plot_engagement_metrics(df):
    df['date'] = pd.to_datetime(df['date'], format='%a %b %d %H:%M:%S %z %Y')
    df.sort_values('date', inplace=True)
    
    # Plot likes and retweets over time
    fig = px.line(df, x='date', y=['like_count', 'retweet_count'],
                  labels={'date': 'Date', 'value': 'Count'},
                  title='Engagement Metrics: Likes and Retweets Over Time',
                  markers=True)
    fig.update_layout(legend_title_text='Metrics')
    
    # Top liked and retweeted posts
    top_liked = df.nlargest(1, 'like_count')
    top_retweeted = df.nlargest(1, 'retweet_count')
    
    # Add top liked and retweeted posts as annotations
    for _, row in top_liked.iterrows():
        fig.add_annotation(x=row['date'], y=row['like_count'],
                           text=f"Top Liked\n{row['like_count']}",
                           showarrow=True, arrowhead=2, ax=0, ay=-40)
    
    for _, row in top_retweeted.iterrows():
        fig.add_annotation(x=row['date'], y=row['retweet_count'],
                           text=f"Top Retweeted\n{row['retweet_count']}",
                           showarrow=True, arrowhead=2, ax=0, ay=-40)
    
    # Save plot to dedicated folder
    fig.write_html(os.path.join(output_dir, 'engagement_metrics.html'))

# Time Series Analysis
def plot_time_series(df):
    df['date'] = pd.to_datetime(df['date'], format='%a %b %d %H:%M:%S %z %Y')
    df.sort_values('date', inplace=True)
    
    fig = px.line(df, x='date', y=['like_count', 'retweet_count'],
                  labels={'date': 'Date', 'value': 'Count'},
                  title='Time Series Analysis: Likes and Retweets Over Time',
                  markers=True)
    fig.update_layout(legend_title_text='Metrics')
    
    # Save plot to dedicated folder
    fig.write_html(os.path.join(output_dir, 'time_series_analysis.html'))

# Plot Topic Classification
def plot_topic_classification(df, topic_names):
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Number of Articles']
    topic_counts['Topic Name'] = topic_counts['Topic'].map(topic_names)
    
    fig = px.bar(topic_counts, x='Topic Name', y='Number of Articles',
                 labels={'Topic Name': 'Topic', 'Number of Articles': 'Number of Articles'},
                 title='Topic Classification Distribution')
    
    # Save plot to dedicated folder
    fig.write_html(os.path.join(output_dir, 'topic_classification.html'))

# Hashtag Analysis
def plot_hashtag_analysis(df):
    hashtags = df['hashtags'].explode().dropna().tolist()
    hashtag_counts = Counter(hashtags)
    
    # Get top 10 hashtags
    top_hashtags = hashtag_counts.most_common(10)
    hashtags, counts = zip(*top_hashtags)
    
    fig = px.bar(x=hashtags, y=counts,
                 labels={'x': 'Hashtags', 'y': 'Frequency'},
                 title='Top 10 Hashtag Frequency Distribution')
    
    # Add hashtags inside the bars
    fig.update_traces(text=hashtags, textposition='inside')
    
    # Save plot to dedicated folder
    fig.write_html(os.path.join(output_dir, 'hashtag_analysis.html'))

# Main script
if __name__ == "__main__":
    df, kmeans, topic_names = classify_topics(df)
    plot_engagement_metrics(df)
    plot_time_series(df)
    plot_topic_classification(df, topic_names)
    plot_hashtag_analysis(df)
    print("Analysis completed and interactive images saved in the 'plots' folder.")
