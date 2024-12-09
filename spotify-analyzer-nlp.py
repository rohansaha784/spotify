import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Database and table configuration
DB_NAME = "spotify_cs210.db"
TABLE_NAME = "spotify_songs"

# 1. Fetch and Preprocess Data
def fetch_data(db_name, table_name):
    conn = sqlite3.connect(db_name)
    query = f"SELECT name, artists, energy, danceability, valence, tempo, lyrics FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.dropna(subset=['energy', 'danceability', 'valence', 'tempo', 'lyrics'], inplace=True)
    return df

# 2. KMeans Clustering for Playlists
def cluster_songs(df):
    # Features for clustering
    features = df[['energy', 'danceability', 'valence', 'tempo']]

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)

    # Visualization of clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df['energy'], df['danceability'], c=df['cluster'], cmap='viridis', alpha=0.7)
    plt.title("Song Clusters")
    plt.xlabel("Energy")
    plt.ylabel("Danceability")
    plt.colorbar(label="Cluster")
    plt.grid()
    plt.show()

    return df

# 3. Sentiment Analysis of Lyrics
def analyze_lyrics_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['lyrics'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Visualize sentiment distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['sentiment_score'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Sentiment Distribution of Lyrics")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Songs")
    plt.grid()
    plt.show()

    return df

# 4. Save Results to Database
def save_results_to_db(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df[['name', 'artists', 'cluster', 'sentiment_score']].to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Results saved to {table_name} in {db_name}")

# Main Execution
def main():
    print("Fetching data...")
    df = fetch_data(DB_NAME, TABLE_NAME)
    
    print("Clustering songs for playlist generation...")
    df = cluster_songs(df)
    
    print("Analyzing lyrics sentiment...")
    df = analyze_lyrics_sentiment(df)
    
    print("Saving results to the database...")
    save_results_to_db(df, DB_NAME, "spotify_analysis_results")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
