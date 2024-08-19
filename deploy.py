import pandas as pd
import streamlit as st
import gzip
import pickle

# Function to load compressed pickle files
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

# Load pickled objects
final_df = load_compressed_pickle('final_df.pkl.gz')
similarity = load_compressed_pickle('similarity.pkl.gz')


# Function to recommend movies
def recommend(movie):
    try:
        # Check if the movie is provided and not empty
        if not movie:
            return ["Please enter a movie title."]
        
        # Find the movie index
        movie_index = final_df[final_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        
        # Get top 5 similar movies
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [final_df.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    
    except IndexError:
        return [f"Movie '{movie}' not found."]
    except Exception as e:
        return [f"An error occurred: {str(e)}"]

# Streamlit app
st.write("Hello this is your personal movie recommendation site\nJust type in the movie name and 5 similar movies will be provided to increase your movie experience")
st.write("Enter The Movie:")
x = st.text_input("Movie (First letter capital, proper space, enter the movie name as it may appear on google)")

if st.button("Recommend"):
    # Call the recommend function
    recommendations = recommend(x)
    
    # # Debugging: Display what's returned by recommend()
    # st.write("Debug info:", recommendations)
    
    # Display the recommendations
    st.write("Here are some recommended movies:")
    for movie in recommendations:
        st.write(f"- {movie}")

# # Test the function
# movie = input("Enter the name of the movie: ")

# print("Here are some recommended movies: \n")
# recommend(movie)
