# 🎬 Movie Recommender System

A content-based movie recommendation system built with **NLP (Natural Language Processing)** techniques — specifically **TF-IDF vectorization** and **cosine similarity** — served via a **FastAPI** backend.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Concepts Explained](#concepts-explained)
  - [What is Count Vectorizer?](#1-what-is-count-vectorizer)
  - [What is TF-IDF?](#2-what-is-tf-idf)
  - [How TF-IDF Improves Count Vectorizer](#3-how-tf-idf-improves-count-vectorizer)
  - [Cosine Similarity](#4-cosine-similarity)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1 — Data Loading & Cleaning](#step-1--data-loading--cleaning)
  - [Step 2 — Feature Engineering (Tags)](#step-2--feature-engineering-tags)
  - [Step 3 — Text Preprocessing (NLP)](#step-3--text-preprocessing-nlp)
  - [Step 4 — TF-IDF Vectorization](#step-4--tf-idf-vectorization)
  - [Step 5 — Index Mapping](#step-5--index-mapping)
  - [Step 6 — Recommendation via Cosine Similarity](#step-6--recommendation-via-cosine-similarity)
  - [Step 7 — Serializing Artifacts](#step-7--serializing-artifacts)
- [Project Structure](#project-structure)
- [FastAPI Backend](#fastapi-backend)
  - [Startup & Model Loading](#startup--model-loading)
  - [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)

---

## Overview

This project implements a **content-based recommendation system** for movies. Given any movie title, the system finds the most similar movies based on the textual content of their **overview**, **genres**, and **tagline**.

The key idea is:

> *Imagine a scenario where a user watched a movie and gave it a high rating. What other movies would you recommend based on what he watched? Movies with the most similar tags and genres are the best choices.*

To achieve this, we convert each movie's description into a vector using **TF-IDF**, and then measure closeness between movies using **cosine similarity**. Similar movies are those whose vectors are closest to each other in high-dimensional space.

---

## Concepts Explained

### 1. What is Count Vectorizer?

For those exposed to NLP, one of the most widely recognized methods for processing text is **converting text into vectors** that can be used in a model. Count Vectorizer is one such method.

**How it works:**

Each document (e.g., a sentence or a movie description) is represented as a vector of word counts. If your vocabulary has `V` unique words, each document becomes a `1×V` vector. With `N` documents, you get an `N×V` matrix.

**Toy Example:**

Consider three documents:
```
'I like apple'
'I like avocado'
'I do not like avocado, but avocado is healthy'
```

The vocabulary (sorted) is: `[apple, avocado, but, do, healthy, I, is, like, not]` — 9 words.

| Document | apple | avocado | but | do | healthy | I | is | like | not |
|---|---|---|---|---|---|---|---|---|---|
| I like apple | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| I like avocado | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| I do not like avocado, but avocado is healthy | 0 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

The three documents are converted into a **3×9 matrix**, where 9 is the vocabulary size.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'I like apple',
    'I like avocado',
    'I do not like avocado, but avocado is healthy'
]
c2v = CountVectorizer()
c2v.fit_transform(corpus)
```

---

### 2. What is TF-IDF?

**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**. It is a technique that improves upon raw Count Vectorization.

The formula has two parts:

**TF (Term Frequency):**
This is essentially the output of the Count Vectorizer — how many times a term `t` appears in document `d`.

```
tf(t, d) = count of term t in document d
```

**IDF (Inverse Document Frequency):**

```
idf(t) = log( N / N(t) )
```

Where:
- `N` = total number of documents
- `N(t)` = number of documents containing term `t`

**TF-IDF score:**
```
tfidf(t, d) = tf(t, d) × idf(t)
```

**Intuition:** If `N/N(t)` gets larger (i.e., the term appears in fewer documents), `idf(t)` rises. This means **rare terms get higher weights**, and **common terms get lower weights** — automatically.

**Example of idf() scaling:**

| Term | Appears in N(t) docs | idf = log(N / N(t)) | Effect |
|---|---|---|---|
| `"the"` | Almost all docs | ≈ 0 (very small) | Shrinks the term's weight |
| `"adventure"` | Few docs | Large value | Amplifies the term's weight |

---

### 3. How TF-IDF Improves Count Vectorizer

The core limitation of Count Vectorizer is **stop words**.

> *Stop words are words that appear everywhere but don't necessarily carry a lot of information — such as "are", "is", "I", "a", "have", "so", "the", etc.*

For example, in spam detection, both spam and non-spam emails contain the word "the". Having a high count for "the" is unhelpful and can even mislead a model. Additionally, stop words **increase the dimensionality** of vectors, which means more computation, more memory, and higher cost.

**The TF-IDF solution:** Stop words appear in almost every document, so their `N(t)` ≈ `N`, making `log(N/N(t)) ≈ 0`. This **automatically scales down** stop words without any hand-crafted list.

> *Stop words are application-specific and change. For example, if articles are about artificial intelligence and physics, the term "machine learning" can be useful. However, if all texts are about AI, "machine learning" is likely a stop word. TF-IDF can crack this problem — the essential idea is to scale down the count of stop words while scaling up the count of meaningful words.*

---

### 4. Cosine Similarity

Once each movie is represented as a TF-IDF vector, we need a way to measure how "close" two movies are.

**Cosine similarity** measures the cosine of the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- A value of **1.0** means the vectors are identical (same direction).
- A value of **0.0** means the vectors are completely unrelated (orthogonal).

This metric is preferred over Euclidean distance for text because it is **length-independent** — a long movie overview and a short one can still score highly similar if they share the same key terms.

---

## Pipeline Walkthrough

The pipeline is defined in [`recommender.ipynb`](./recommender.ipynb) (and mirrored in the standalone script). Here is a step-by-step breakdown:

### Step 1 — Data Loading & Cleaning

```python
df = pd.read_csv("movies_metadata.csv")

# Drop duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Keep only relevant columns
df = df[['title', 'overview', 'genres', 'tagline', 'vote_average', 'popularity']]

# Drop rows with no title (can't recommend something with no name)
df = df.dropna(subset=['title'])

# Fill missing overviews and taglines with empty strings
df['overview']  = df['overview'].fillna(' ')
df['tagline']   = df['tagline'].fillna(' ')
```

**Why?**
The raw dataset (`movies_metadata.csv` from Kaggle) contains 24 columns, many of which are irrelevant (e.g., `budget`, `revenue`, `adult`). We narrow down to the columns that carry **descriptive textual meaning** about the movie. Missing values in `overview` and `tagline` are filled with spaces rather than dropped, because we still want those movies to be recommendable.

The `genres` column is stored as a JSON-like string (`"[{'id': 16, 'name': 'Animation'}, ...]"`), so it is parsed and flattened:

```python
df['genres'] = df['genres'].apply(
    lambda x: " ".join([i['name'] for i in ast.literal_eval(x)])
)
# Result: "Animation Comedy Family"
```

---

### Step 2 — Feature Engineering (Tags)

```python
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']
```

**Why?**
No single column fully describes a movie. The `overview` tells the story, `genres` tells the category, and `tagline` is a catchy phrase that captures the movie's essence. Concatenating all three into a single `tags` column gives us a **rich, combined description** that the TF-IDF model can learn from.

**Example output for row 45:**
```
Soon-to-be-wed graduate student Finn Dodd develops cold feet when she suspects
her fiancé is cheating on her... Drama Romance There's beauty in the patterns of life.
```

---

### Step 3 — Text Preprocessing (NLP)

Before vectorizing, we clean the raw text to reduce noise and normalize the vocabulary.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Lowercase everything
    text = str(text).lower()

    # 2. Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenize (split into individual words)
    words = text.split()

    # 4. Remove stop words
    words = [word for word in words if word not in stop_words]

    # 5. Lemmatize (reduce words to their root form)
    words = [lemmatizer.lemmatize(word) for word in words]

    # 6. Reassemble into a single string
    return " ".join(words)

df['tags'] = df['tags'].apply(preprocess_text)
```

**Each step explained:**

| Step | What it does | Example |
|---|---|---|
| **Lowercase** | Normalizes case so "Action" and "action" are the same token | `"Action"` → `"action"` |
| **Remove punctuation** | Strips commas, apostrophes, hyphens, etc. | `"can't"` → `"cant"` |
| **Tokenize** | Splits a string into a list of words | `"action film"` → `["action", "film"]` |
| **Remove stop words** | Drops common, non-informative words | `["a", "the", "is"]` → dropped |
| **Lemmatize** | Reduces words to their dictionary base form | `"running"` → `"run"`, `"wolves"` → `"wolf"` |

**Why lemmatize instead of stemming?** Lemmatization uses a vocabulary and morphological analysis, producing proper words. Stemming is faster but can produce non-words (e.g., `"running"` → `"runn"`). Lemmatization gives cleaner vocabulary for TF-IDF.

---

### Step 4 — TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_df=50000,        # Ignore terms that appear in more than 50,000 documents
    ngram_range=(1, 2),  # Consider single words AND two-word phrases (bigrams)
    stop_words='english' # An additional built-in stop-word filter
)

tfidf_matrix = tfidf.fit_transform(df['tags'])
# Output: <Compressed Sparse Row matrix of shape (45447, 1068907)>
```

**Parameters explained:**

| Parameter | Value | Purpose |
|---|---|---|
| `max_df` | `50000` | Terms appearing in more than 50,000 documents are likely stop words and are excluded |
| `ngram_range` | `(1, 2)` | Captures both unigrams (`"action"`) and bigrams (`"action adventure"`), preserving multi-word phrases |
| `stop_words` | `'english'` | Applies sklearn's built-in English stop-word list as a secondary filter |

**Result:** A sparse matrix of shape **(45,447 movies × 1,068,907 features)**. Each row is a movie; each column is a unique term or bigram. The values are TF-IDF scores — high scores indicate important, distinctive terms for that movie.

> The matrix is stored as a **Compressed Sparse Row (CSR)** matrix because the vast majority of entries are zero (most words don't appear in most movies). CSR format saves significant memory.

---

### Step 5 — Index Mapping

```python
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
```

**Why?** To look up a movie by title and get its row number in the TF-IDF matrix, we create a reverse mapping: `title → row index`. Duplicate titles are dropped (keeping the first occurrence).

**Output (sample):**
```
title
Toy Story                          0
Jumanji                            1
Grumpier Old Men                   2
...
Queerama                       45446
Length: 45447, dtype: int64
```

---

### Step 6 — Recommendation via Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend(title, n=10):
    if title not in indices:
        return ["Movie not found"]

    # Get the row index of the queried movie
    idx = indices[title]

    # Compute cosine similarity between this movie and ALL other movies
    similarity_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort by similarity score descending; skip index 0 (the movie itself)
    similar_index = similarity_score.argsort()[::-1][1:n+1]

    return df['title'].iloc[similar_index]
```

**Step-by-step:**

1. **Lookup** — Convert the title string to its integer row index using the `indices` map.
2. **Query vector** — Extract the TF-IDF row vector for that movie (`tfidf_matrix[idx]`).
3. **Similarity** — Compute cosine similarity between the query vector and every other row in the matrix.
4. **Sort** — Sort all similarity scores descending (`argsort()[::-1]`).
5. **Skip self** — Skip index `[0]` because the most similar movie to "Toy Story" is "Toy Story" itself.
6. **Return top-N** — Return the titles of the top `n` most similar movies.

**Example output for `recommend('Toy Story')`:**
```
Toy Story 2
Toy Story 3
Superstar Goofy
Small Fry
What's Up, Tiger Lily?
...
```

---

### Step 7 — Serializing Artifacts

After training, all model artifacts are serialized (pickled) to disk so the FastAPI server can load them at startup without retraining:

```python
import pickle

pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))  # The TF-IDF sparse matrix
pickle.dump(indices,      open('indices.pkl',      'wb'))  # The title → index mapping
df.to_pickle('df.pkl')                                      # The cleaned DataFrame
pickle.dump(tfidf,        open('tfidf.pkl',        'wb'))  # The fitted vectorizer object
```

**Why pickle?** These artifacts (especially `tfidf_matrix` with ~2.7 million stored elements) are expensive to recompute. Serializing them means the API server starts instantly by simply loading pre-computed data.

---

## Project Structure

```
Movie-Recommender-System/
│
├── recommender.ipynb       # Jupyter notebook: full training pipeline
├── import numpy as np.py   # Standalone Python script version of the pipeline
├── main.py                 # FastAPI backend server
│
├── df.pkl                  # Serialized cleaned DataFrame
├── indices.pkl             # Serialized title → index mapping (pandas Series)
├── tfidf_matrix.pkl        # Serialized TF-IDF sparse matrix (45447 × 1068907)
├── tfidf.pkl               # Serialized fitted TfidfVectorizer object
│
├── .env                    # Environment variables (TMDB_API_KEY) — not committed
└── README.md
```

---

## FastAPI Backend

[`main.py`](./main.py) serves the recommendation engine as a REST API, enriching results with live movie data from the **TMDB (The Movie Database) API**.

### Startup & Model Loading

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open("df.pkl",           "rb") as f: df           = pickle.load(f)
    with open("indices.pkl",      "rb") as f: indices_obj  = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f: tfidf_matrix = pickle.load(f)
    with open("tfidf.pkl",        "rb") as f: tfidf_obj    = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    yield
```

All four pickle files are loaded into memory once when the server starts. The `TITLE_TO_IDX` dictionary normalizes all titles to lowercase for case-insensitive lookups.

### Recommendation Logic in the API

Instead of using sklearn's `cosine_similarity` (which materializes a full N×N matrix), the API uses a dot-product approach for efficiency:

```python
qv = tfidf_matrix[idx]                            # query vector (1 × V)
scores = (tfidf_matrix @ qv.T).toarray().ravel()  # dot product with all rows
order = np.argsort(-scores)                        # sort descending
```

Since TF-IDF vectors are L2-normalized by sklearn, the dot product equals cosine similarity — but this approach avoids allocating a huge intermediate matrix, making it much faster for large datasets.

---

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/home` | Browse movies by category (`popular`, `top_rated`, `trending`, `upcoming`, `now_playing`) |
| `GET` | `/tmdb/search` | Search TMDB for movies by keyword |
| `GET` | `/movie/id/{tmdb_id}` | Get full details of a movie by TMDB ID |
| `GET` | `/recommend/tfidf` | Get TF-IDF content-based recommendations by title |
| `GET` | `/recommend/genre` | Get genre-based recommendations via TMDB Discover API |
| `GET` | `/movie/search` | **Bundle endpoint** — returns movie details + TF-IDF recs + genre recs in one call |

#### `/recommend/tfidf`
```
GET /recommend/tfidf?title=Avatar&top_n=10
```
Returns the top-N most similar movies from the local TF-IDF dataset, with similarity scores.

#### `/movie/search` (Bundle)
```
GET /movie/search?query=The+Dark+Knight&tfidf_top_n=12&genre_limit=12
```
This is the primary endpoint for a frontend. It:
1. Searches TMDB for the best matching movie
2. Fetches full movie details (poster, overview, genres, etc.)
3. Runs TF-IDF content-based recommendations from the local dataset
4. Fetches genre-based recommendations from TMDB's Discover API
5. Enriches each TF-IDF result with a TMDB poster and metadata
6. Returns everything in a single, structured `SearchBundleResponse`

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 1. Install dependencies

```bash
pip install fastapi uvicorn httpx pandas numpy scikit-learn nltk python-dotenv
```

### 2. Configure environment

Create a `.env` file in the project root:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

### 3. Run the training notebook

Open `recommender.ipynb` in Jupyter and run all cells. This will generate:
- `df.pkl`
- `indices.pkl`
- `tfidf_matrix.pkl`
- `tfidf.pkl`

### 4. Start the API server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Dataset

The dataset used is the **TMDB Movie Metadata** dataset available on Kaggle:

- [TMDB Movie Metadata — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- File used: `movies_metadata.csv`
- Contains metadata for **~45,000 movies** including overviews, genres, taglines, ratings, and more.

---

> *"NLP is a most useful tool to create a recommendation system. Although the NLP theories introduced here are the simplest, they can be used to build a surprisingly effective recommendation system."*
