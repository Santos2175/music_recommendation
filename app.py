from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from keras.models import load_model

app = Flask(__name__, static_url_path='/static')

# Load the model
model = load_model(
    '/Users/santo/OneDrive/Desktop/song recommendation system/model/music_recommend.h5')

# Load the dataset containing user IDs, song IDs, and ratings
data = pd.read_csv(os.path.join('input', 'merged_dataset.csv'))

# Determine the total number of songs and users
num_songs = data['music_id'].nunique()
num_users = data['user_id'].nunique()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = int(request.form['user_id'])

    # Validate the user ID
    if user_id < 1 or user_id > num_users:
        error_message = "User ID not available"
        return render_template('error.html', error_message=error_message)

    # Retrieve the songs listened to by the user
    user_songs = data[data['user_id'] == user_id]['music_id'].unique()

    # Find unrated songs
    user_unrated_songs = np.setdiff1d(np.arange(1, num_songs + 1), user_songs)

    # Create user input array for the unrated songs
    user_input = np.repeat(user_id, len(user_unrated_songs))

    # Predict ratings for the unrated songs
    predicted_ratings = model.predict(
        [user_input, user_unrated_songs]).flatten()

    # Sort the predicted ratings and get the indices of the top recommended songs
    recommended_song_indices = np.argsort(predicted_ratings)[::-1][:12]

    # Get the actual song IDs of the recommended songs
    recommended_songs = user_unrated_songs[recommended_song_indices]

    # Get the actual song names of the recommended songs
    recommended_songs = data.loc[data['music_id'].isin(
        user_unrated_songs), 'music_name'].values[recommended_song_indices]

    return render_template('recommendations.html', user_id=user_id, recommended_songs=recommended_songs)


if __name__ == '__main__':
    app.run()
