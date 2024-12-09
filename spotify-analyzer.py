import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Database configuration
db_name = "spotify_cs210.db"
table_name = "spotify_cs210"

# Fetch and preprocess data
def fetch_and_prepare_data(db_name, table_name):
    conn = sqlite3.connect(db_name)
    query = f"SELECT energy, danceability, valence, instrumentalness, loudness, duration_ms, tempo, country FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Drop missing values
    df.dropna(inplace=True)

    # One-hot encode categorical columns (e.g., 'country')
    df = pd.get_dummies(df, columns=['country'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    return pd.DataFrame(scaled_features, columns=df.columns)

# Predict genre
def predict_genre(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Print performance metrics
    print("Genre Prediction - Random Forest Classifier")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cluster songs
def cluster_songs(X):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title("Clusters of Songs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster")
    plt.grid()
    plt.show()

    return clusters

# Save results to the database
def save_results_to_db(db_name, table_name, results, column_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Add the new column for results if it doesn't exist
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} INTEGER")
    conn.commit()

    # Insert the results
    query = f"UPDATE {table_name} SET {column_name} = ? WHERE ROWID = ?"
    for idx, value in enumerate(results, start=1):
        cursor.execute(query, (int(value), idx))
    conn.commit()
    conn.close()

# Main function to execute all tasks
def main():
    print("Fetching and preparing data...")
    df = fetch_and_prepare_data(db_name, table_name)

    if 'genre' in df.columns:
        X = df.drop(columns=['genre'])
        y = df['genre']
        print("Predicting genres...")
        predict_genre(X, y)
    else:
        print("Clustering songs...")
        clusters = cluster_songs(df.values)
        print("Saving clustering results to database...")
        save_results_to_db(db_name, table_name, clusters, "cluster")

if __name__ == "__main__":
    main()
