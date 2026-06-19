import numpy as np 
import ast
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



df = pd.read_csv("E:\\Project\\Recommender System\\movies_metadata.csv")


df = df.drop_duplicates().reset_index(drop = True)
df = df[['title', 'overview', 'genres', 'tagline', 'vote_average', 'popularity']]
# remove the blank titles:
df = df.dropna(subset=['title'])

# fill blank overview fill blank space:
df['overview'] = df['overview'].fillna(' ')

# df.iloc[0]['genres']      => "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]"
df['genres'] = df['genres'].apply(lambda x:" ".join([i['name'] for i in ast.literal_eval(x)]))

# df['tagline'].isnull().sum()      => 25000+ are null
df['tagline'] = df['tagline'].fillna(' ')

# details:
df.isnull().sum()df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']
nltk.download('stopwords')
nltk.download('wordnet')stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # lower case:
    text = str(text).lower()
    
    # remove punctuation:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # tokenize words:
    words = text.split()
    
    # remove stop words (Fixed the loop variable here):
    words = [word for word in words if word not in stop_words]
    
    # lemmatize:
    words = [lemmatizer.lemmatize(word) for word in words]

    # reassemble:
    return " ".join(words)

df['tags'] = df['tags'].apply(preprocess_text)

    # reset the index
df = df.reset_index(drop = True)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# apply TFIDF
tfidf = TfidfVectorizer(max_df=50000, ngram_range=(1,2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])


def recommend(title, n=10):    # for 10 movies recommendation
    if title not in indices:
        return ["Movie not fount"]
    idx = indices[title]
    similarity_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_index = similarity_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_index]

recommend('Toy Story')