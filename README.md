# Spotify Song Recommendation System - README

This project demonstrates a personalized song recommendation system using Spotify's API. It allows users to fetch song data from their Spotify account, generate a dataset, and predict song recommendations based on user-provided descriptions.

---

## Prerequisites

1. **Python Environment**: Make sure you have Python 3.x installed.

2. **Required Libraries**: Install the following libraries:
   - `spotipy`
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `sqlite3`

   Use the following command to install the required libraries:
   ```bash
   pip install spotipy pandas numpy matplotlib scikit-learn
   ```

3. **Spotify Developer Account**:
   - Visit the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/) and create an app.
   - Note down your **Client ID**, **Client Secret**, and set a **Redirect URI** (e.g., `http://localhost:8888/callback`).

---

## Usage Instructions

### Step 1: Fetch Data using `spotify-data.ipynb`

1. **Setup**:
   - Open the `spotify-data.ipynb` notebook.
   - Replace the placeholder values (`xxx`) in the following section with your Spotify **Client ID**, **Client Secret**, and **Redirect URI**:
     ```python
     cli_id = 'your_client_id_here'
     cli_secret = 'your_client_secret_here'
     url = 'your_redirect_uri_here'
     ```

2. **Run the Notebook**:
   - Execute the cells in the notebook to:
     - Authenticate with Spotify.
     - Fetch your top songs and their audio features.
   - The notebook will save a CSV file (i.e., `data/Varun-Filtered.csv`) containing the fetched data.

---

### Step 2: Run Predictions using `main.ipynb`

1. **Setup**:
   - Make sure that the CSV file generated from `spotify-data.ipynb` is available in the `data` folder.
   - Open the `main.ipynb` notebook.

2. **Run the Notebook**:
   - Execute all cells in the `main.ipynb` notebook.
   - The notebook will:
     - Load the song dataset into an SQLite database.
     - Use user input (i.e., song descriptions) to predict audio features.
     - Return a list of 5 songs that closely match the input.

3. **Input Example**:
   When prompted, provide a description of the desired song, such as:
   ```
   "high energy, danceable, positive vibe"
   ```
   The notebook will display the 5 closest matching songs based on their audio features.
