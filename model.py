import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Reading csv files into dataframe
dfm = pd.read_csv('tmdb_5000_movies.csv')
dfc = pd.read_csv('tmdb_5000_credits.csv')

# Merging the two datasets
df = dfm.merge(dfc, on='title')

# Columns to keep
df = df[['genres', 'id', 'keywords', 'title', 'overview', 'cast', 'crew']]

# Drop null values
df.dropna(inplace=True)

# Converting dictionary to list
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)

def convert_3(obj):
    L = []
    ct = 0 
    for i in ast.literal_eval(obj):
        if ct !=3:
            L.append(i['name'])
            ct += 1
        else:
            break
    return L

df['cast'] = df['cast'].apply(convert_3)

# Function to fetch director
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df['crew'] = df['crew'].apply(fetch_director)

# Rename column
df = df.rename(columns={'crew': 'Director'})

# Converting overview to list
df['overview'] = df['overview'].apply(lambda x: x.split())

# Removing spaces between words
df['genres'] = df['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
df['keywords'] = df['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
df['overview'] = df['overview'].apply(lambda x: [i.replace(' ', '') for i in x])
df['cast'] = df['cast'].apply(lambda x: [i.replace(' ', '') for i in x])

# Making a tag
df['tags'] = df['genres'] + df['keywords'] + df['overview'] + df['cast'] + df['Director']

# Convert list of tags into a string
df['tags'] = df['tags'].apply(lambda x: " ".join(x))

# Convert all uppercase letters into lowercase
df['tags'] = df['tags'].apply(lambda x: x.lower())

# Initialize stemmer
ps = PorterStemmer()

# Stemming
df['tags'] = df['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))

# Making vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags'])

# Save the CountVectorizer
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Save the similarity matrix
with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity, f)

def recommend(movie):
    try:
        movie_index = df[df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        for i in movies_list:
            print(df.iloc[i[0]]['title'])
            
    except IndexError:
        print(movie, 'not found')

# Example usage
recommend('Avatar')
