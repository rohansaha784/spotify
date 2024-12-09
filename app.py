import streamlit as st
import pandas as pd
from spotify_analyzer import fetch_data, cluster_songs, analyze_lyrics_sentiment, save_results_to_db

# Streamlit Configuration
st.set_page_config(page_title="Spotify Analyzer", layout="wide")

# Database and Table Configurations
DB_NAME = "spotify_cs210.db"
TABLE_NAME = "spotify_songs"
OUTPUT_TABLE = "spotify_analysis_results"

# Step 1: User Inputs
st.title("Spotify Analyzer")
st.sidebar.header("Input API Details")
api_key = st.sidebar.text_input("Enter your Spotify API Key")
api_secret = st.sidebar.text_input("Enter your Spotify API Secret", type="password")
redirect_uri = st.sidebar.text_input("Redirect URI", value="http://localhost:8080")
run_ml = st.sidebar.button("Run Analysis")

# Step 2: Load and Display Initial Data
st.header("Loaded Data")
if st.sidebar.button("Load Data"):
    try:
        data = fetch_data(DB_NAME, TABLE_NAME)
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Step 3: Visualizations and Analysis
if run_ml:
    st.header("Clustering Songs for Playlists")
    try:
        data = fetch_data(DB_NAME, TABLE_NAME)
        clustered_data = cluster_songs(data)
        st.subheader("Clustered Data")
        st.dataframe(clustered_data)

        st.header("Sentiment Analysis of Lyrics")
        analyzed_data = analyze_lyrics_sentiment(clustered_data)
        st.subheader("Sentiment Scores")
        st.dataframe(analyzed_data[['name', 'sentiment_score']])

        save_results_to_db(analyzed_data, DB_NAME, OUTPUT_TABLE)
        st.success("Analysis complete! Results saved to the database.")
    except Exception as e:
        st.error(f"Error running analysis: {e}")

# Step 4: Visualization Section
st.header("Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Cluster Visualization")
    if st.button("Show Clusters"):
        try:
            cluster_songs(data)
        except Exception as e:
            st.error(f"Error visualizing clusters: {e}")
with col2:
    st.subheader("Sentiment Distribution")
    if st.button("Show Sentiment Distribution"):
        try:
            analyze_lyrics_sentiment(data)
        except Exception as e:
            st.error(f"Error visualizing sentiment: {e}")
