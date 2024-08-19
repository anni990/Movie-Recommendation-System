import pandas as pd
import streamlit as st
import gzip
import pickle

# Inject CSS to change the background color and style
st.markdown(
        """
        <style>
        .stApp {
            background-color: #1E90FF;  
            color: #FFFFFF;  
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FFFFFF;  
        }

        .stTextInput, .stNumberInput, .stButton>button {
            background-color: #A9A9A9;  
            color: #000000;  
            border: 2px solid #A9A9A9;
            border-radius: 10px;  
        }
        .stMarkdown ul {
            list-style-type: none;
            padding-left: 0;
        }
        .stMarkdown li {
            padding: 5px 0;
            color: #FFFFFF;  
        }
        /* Custom CSS for feedback section */
        .feedback-section {
            margin-bottom: 100px;  /* Adjust the margin as needed */
        }
        .stButton>button:hover {
        color: #1E90FF;  /* Change text color to blue on hover */
        border-color: #1E90FF; /* Change border color to blue on hover */
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# Function to load compressed pickle files
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

# Load pickled objects
final_df = load_compressed_pickle('final_df.pkl.gz')
similarity = load_compressed_pickle('similarity.pkl.gz')

# Function to recommend movies
def recommend(movie, n):
    try:
        # Check if the movie is provided and not empty
        if not movie:
            return ["Please enter a movie title."]
        
        # Find the movie index
        movie_index = final_df[final_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        
        # Get top 'n' similar movies
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
        recommended_movies = [final_df.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    
    except IndexError:
        return [f"Movie '{movie}' not found."]
    except Exception as e:
        return [f"An error occurred: {str(e)}"]

# Streamlit app
st.title("***Movie Recommendation Site*** 😍")
st.header("Welcome to your personal movie recommendation site 🎬")
st.write("Type in the name of a movie, and we'll provide you with similar recommendations!")

x = st.text_input("Movie Name (e.g., 'Inception')", placeholder="Enter movie name here...")
number = st.number_input("Number of recommendations", min_value=1, step=1, value=5)
st.write(f"Number of movies to be predicted: {number}")

if st.button("Recommend"):
    # Call the recommend function
    recommendations = recommend(x, number)
    
    # Display the recommendations
    st.write("Here are some recommended movies 🎬:")
    for movie in recommendations:
        st.write(f"- {movie}")
            
# Feedback section with custom CSS class
st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
# Sentiment mapping
sentiment_mapping = [":thumbs_down:", ":thumbs_up:"]
feedback = st.radio("Give your feedback:", ["👍 Thumbs Up", "👎 Thumbs Down"], index=0)

# Mapping the selection to sentiment
if feedback == "👍 Thumbs Up":
    st.write("You selected: 👍")
    st.write("Thanks for your feedback! 😊")
elif feedback == "👎 Thumbs Down":
    st.write("You selected: 👎")
    st.write("Thanks for your feedback! 😊 We will try to improve. 🤔")


# # Test the function
# movie = input("Enter the name of the movie: ")

# print("Here are some recommended movies: \n")
# recommend(movie)
