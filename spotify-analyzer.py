# spotify_analyzer.py
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk

nltk.download('vader_lexicon')

# Fetch Data from Database
def fetch_data(db_name, table_name):
    conn = sqlite3.connect(db_name)
    query = f"SELECT name, artists, energy, danceability, valence, tempo, lyrics FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.dropna()

# Cluster Songs
def cluster_songs(df):
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['energy', 'danceability', 'valence', 'tempo']])
    return df

# Analyze Sentiment
def analyze_lyrics_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['lyrics'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

# Save Results to Database
def save_results_to_db(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df[['name', 'artists', 'cluster', 'sentiment_score']].to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
