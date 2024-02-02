from flask import Flask, jsonify, request, render_template
import pickle
from fuzzywuzzy import fuzz,process
from flask_cors import CORS
import tensorflow as tf
import gzip
import pandas as pd

def create_app():
    
# Load the compressed model file
    loaded_model = tf.keras.models.load_model('path_to_your/model.h5')
    # Load the pretrained model
    model = loaded_model

    # Load the movie data (assuming you have it stored in a DataFrame)
    movie_data = pd.read_csv('movies_list.csv')

    app = Flask(__name__)
    CORS(app)

    # Replace this with your recommendation function or model
    def get_movie_recommendations(input_title,num_recommendations=5):
        # Preprocess the input movie title (if needed)
        # Here, we'll convert it to lowercase for case-insensitive matching
        input_title_lower = input_title.lower()

        # Find the best match for the input title in the movie data
        best_match = None
        highest_score = 0
        for title in movie_data["original_title"]:
            score = fuzz.ratio(input_title_lower, title.lower())
            if score > highest_score:
                best_match = title
                highest_score = score

        # Get the index of the best match in the movie data
        index = movie_data[movie_data["original_title"] == best_match].index
        
        # Check if a match is found
        if len(index) == 0:
            return []

        # Get recommendations using the pretrained model
        recommendations_indices = model.predict([index])  # Assuming the model expects movie indices as input

        # Sort the recommendation indices based on their scores
        sorted_indices = recommendations_indices.argsort()[0]

        # Convert recommendation indices to movie titles
        recommendations = []
        for idx in sorted_indices[:num_recommendations]:
            if idx < len(movie_data):
                movie_title = movie_data.loc[int(idx), "original_title"]
                recommendations.append(movie_title)

        # Return top num_recommendations recommended movie titles
        return recommendations


    @app.route('/recommendations', methods=['GET'])
    def recommendations():
        # Get the user_id from the query parameter
        movie_title = request.args.get('movie_title')

        if not movie_title:
            return jsonify({"error": "Movie Title is required"}), 400

        # Call your recommendation function or model
        recommended_movies = get_movie_recommendations(movie_title)

        # Return the recommended movies as JSON
        return jsonify({"recommendations": recommended_movies})
    
    return app



      
    