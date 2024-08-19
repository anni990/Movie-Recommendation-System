import pandas as pd
import numpy as np
import ast
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # Download nltk data if not already downloaded
# nltk.download('punkt')

# Read data
dfm = pd.read_csv('tmdb_5000_movies.csv')
dfc = pd.read_csv('tmdb_5000_credits.csv')

# Merge dataframes
df = dfm.merge(dfc, on='title')
df = df[['genres', 'id', 'keywords', 'title', 'overview', 'cast', 'crew']]
df.dropna(inplace=True)

# Convert string representations of lists to lists
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
        if ct != 3:
            L.append(i['name'])
            ct = ct + 1
        else:
            break
    return L

df['cast'] = df['cast'].apply(convert_3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df['crew'] = df['crew'].apply(fetch_director)
df = df.rename(columns={'crew': 'Director'})

# Preprocess text data
ps = PorterStemmer()
df['overview'] = df['overview'].apply(lambda x: x.split())
df['genres'] = df['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
df['keywords'] = df['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
df['overview'] = df['overview'].apply(lambda x: [i.replace(' ', '') for i in x])
df['cast'] = df['cast'].apply(lambda x: [i.replace(' ', '') for i in x])

# Combine tags
df['tags'] = df['genres'] + df['keywords'] + df['overview'] + df['cast'] + df['Director']
final_df = df[['id', 'title', 'tags']]

# Lowercase and stem tags
final_df['tags'] = final_df['tags'].apply(lambda x: " ".join(x))
final_df['tags'] = final_df['tags'].apply(lambda x: x.lower())

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

final_df['tags'] = final_df['tags'].apply(stem)

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(final_df['tags']).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Save final_df and similarity to pickle files
final_df.to_pickle('final_df.pkl')
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

# Function to recommend movies
def recommend(movie):
    try:
        movie_index = final_df[final_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        for i in movies_list:
            print(final_df.iloc[i[0]].title)
    except IndexError:
        print(movie + ' not found')

# Test the function
print(recommend('Iron Man 2'))
