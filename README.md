🎬 Production Movie Recommender System

A hybrid, content-based movie recommendation platform built with NLP, TF-IDF, FastAPI, the TMDB REST API, Streamlit, and SQLite.

📑 Table of Contents

Executive Summary & High-Level Architecture

Phase 1: Data Ingestion & Exploratory Data Analysis (EDA)

Phase 2: Feature Engineering & NLP Preprocessing Pipeline

Phase 3: Mathematical Vector Space Modeling (TF-IDF & CSR Matrices)

Phase 4: Similarity Metric Derivation & Ranking Mechanics

Phase 5: Asynchronous Microservice Layer (FastAPI)

Phase 6: Frontend Architecture & Dynamic Routing (Streamlit)

Phase 7: Behavioral Analytics & Relational Telemetry (SQLite)

Installation & Operational Guide

Technical Glossaries & Concept Reference

1. Overview & System Architecture

This project uses a content-based recommendation pipeline with real-time movie information and user interaction tracking.

End-to-End System Workflow

[ Raw CSV: 45k+ Movies ] │ ▼ [ Phase 1: EDA & Cleaning ] ─── Null Handling, Deduplication, JSON Parsing │ ▼ [ Phase 2: NLP Pipeline ] ───── Case Folding, Regex, Stopwords, WordNet Lemmatization │ ▼ [ Phase 3: Vectorization ] ──── TF-IDF (Unigram + Bigram) ──> Compressed Sparse Row (CSR) │ ▼ [ Phase 4: Model Serialization ] ── Pickle: tfidf_matrix.pkl, indices.pkl, df.pkl │ ├───────────────────────────────────────────┐ ▼ ▼ [ Phase 5: FastAPI Backend ] [ TMDB REST API ]

Lifespan in-memory loading - Posters & Backdrops

O(1) Index Reverse-Lookup - Real-time Trend Feeds

On-the-fly Dot-Product Ranking - Genre Discovery │ ▼ HTTP JSON API [ Phase 6: Streamlit UI ] ────────────────── Autocomplete Search, Poster Grids │ ▼ Session Actions (Search / Click / Watch) [ Phase 7: SQLite DB (user_history.db) ] ─── Demographics & User Activity Logs

---

Phase 1: Data Ingestion & EDA

The foundation of the recommendation engine relies on the Movies Metadata Dataset (`movies_metadata.csv`), containing metadata for 45,466 feature films.

1. Data Audit & Null Analysis

The initial data analysis showed missing values in several important columns:

Total rows: `45,466`

`overview`: `954` missing values

`tagline`: `25,054` missing values (~55% sparsity)

`title`: `6` missing values

2. Cleaning & Sanitization Procedures

Deduplication: Repeated movie instances were identified and pruned using `df.drop_duplicates()`, ensuring index stability.

Subset Selection: The dataset was trimmed down from 24 columns to the core informational signals:

$$\text{Columns} = \{\text{title}, \text{overview}, \text{genres}, \text{tagline}, \text{vote\_average}, \text{popularity}\}$$

Missing Value Imputation:

Rows missing a `title` were dropped entirely, as titles serve as the primary human-readable reference key.

Missing entries in `overview` and `tagline` were imputed with empty strings (`' '`) rather than sentinel tokens (e.g., `"unknown"`), preventing synthetic noise from contaminating the vocabulary during vectorization.

Structured JSON Parsing:

The `genres` attribute in the raw dataset is formatted as a serialized JSON string:

```json

"[{\"id\": 16, \"name\": \"Animation\"}, {\"id\": 35, \"name\": \"Comedy\"}]"

Using Python’s ast.literal_eval, this was extracted into clean, space-delimited string representations: $$\text{"Animation Comedy Family"}$$

Phase 2: Feature Engineering & NLP Preprocessing

To compare movies based on their content, the metadata is combined into one text field and then cleaned and normalized.

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐

│     Overview     │  +  │      Genres      │  +  │     Tagline      │

│  (Plot summary)  │     │ (Category tags)  │     │(Marketing pitch) │

└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘

     │                        │                        │

     └────────────────────────┼────────────────────────┘

                              ▼

                ┌───────────────────────────┐

                │      Composite 'tags'     │

                └─────────────┬─────────────┘

                              ▼

                ┌───────────────────────────┐

                │     Text Normalization    │

                │   - Lowercase folding     │

                │   - Regex punctuation     │

                └─────────────┬─────────────┘

                              ▼

                ┌───────────────────────────┐

                │   Stop-Word Filtering     │

                │   (NLTK English Corpus)   │

                └─────────────┬─────────────┘

                              ▼

                ┌───────────────────────────┐

                │   WordNet Lemmatization   │

                │  (Morphological Roots)    │

                └───────────────────────────┘

1. Composite Document Creation (tags)

A single feature string is created by combining the plot, genres, and tagline: $$\text{tags} = \text{overview} \parallel \text{" "} \parallel \text{genres} \parallel \text{" "} \parallel \text{tagline}$$

Engineering Rationale: An overview explains the story (e.g., "space exploration, black hole"), genres provide categorical grouping (e.g., "Adventure Drama Sci-Fi"), and taglines highlight tone (e.g., "Mankind was born on Earth. It was never meant to die here."). Combining them creates a comprehensive semantic fingerprint for each film.

2. Text Normalization

Case Folding: All characters are converted to lowercase. This ensures identical tokens with differing capitalization (e.g., "Alien", "alien") map to the exact same vector dimension.

Regex Sanitization: Non-alphabetical characters and punctuation are stripped using regular expressions:

python

text = re.sub(r'[^a-zA-Z\s]', '', text)

This eliminates punctuation artifacts ("super-hero", "superhero.") while preserving whitespace boundaries.

3. Stop-Word Removal

Grammatical function words (e.g., "the", "and", "in", "at") appear frequently across almost every movie description, contributing no discriminative semantic power. These are removed using NLTK’s standard English stop-word vocabulary.

4. Morphological Analysis: Lemmatization vs. Stemming

Rather than employing naive rule-based stemming (e.g., Porter or Snowball Stemmers), the pipeline implements WordNet Lemmatization:

Stemming: Heuristically chops word affixes (e.g., "universe" $\rightarrow$ "univers", "flying" $\rightarrow$ "fli"), often producing non-linguistic substrings that degrade interpretability.

Lemmatization: Uses a complete lexical database (WordNet) to analyze the morphological structure of words, transforming inflections into their canonical dictionary headword (lemma): $$\text{"destabilized"} \rightarrow \text{"destabilize"}, \quad \text{"superheroes"} \rightarrow \text{"superhero"}$$

Phase 3: TF-IDF & Vector Space Modeling

Raw text cannot be compared directly, so it is converted into numerical vectors.

1. Term Frequency ($TF$)

Term Frequency measures how frequently a specific term $t$ appears inside document $d$: $$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$ Where $f_{t, d}$ represents the raw occurrence count of term $t$ in document $d$. Dividing by the document length normalizes the metric, preventing long overviews from artificially inflating term weights.

2. Inverse Document Frequency ($IDF$)

If a word appears in every film synopsis across the entire dataset (e.g., "movie", "story"), it holds low informational value for distinguishing between movies. $IDF$ quantifies the rarity of term $t$ across the corpus $D$: $$IDF(t, D) = \log\left(\frac{1 + |D|}{1 + |{d \in D : t \in d}|}\right) + 1$$

$|D|$ is the total number of documents ($45,447$).

$|{d \in D : t \in d}|$ is the document frequency ($df(t)$) — the count of documents containing term $t$.

The constant $+1$ in the numerator and denominator prevents division-by-zero errors (Laplace smoothing).

3. The Combined $TF\text{-}IDF$ Score

The weight assigned to term $t$ in document $d$ is: $$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

Following vectorization, each document vector $\mathbf{v}$ is scaled via $L_2$ Euclidean Normalization: $$\mathbf{v}{\text{norm}} = \frac{\mathbf{v}}{|\mathbf{v}|2} = \frac{\mathbf{v}}{\sqrt{\sum{i=1}^{M} v_i^2}}$$ This guarantees that all document vectors have an identical length of $1$ ($|\mathbf{v}{\text{norm}}|_2 = 1$).

4. N-Gram Modeling (ngram_range=(1, 2))

To preserve phrase-level context that single words (unigrams) lose, the vectorizer incorporates bigrams:

Unigrams: ["star"], ["wars"], ["science"], ["fiction"]

Bigrams: ["star wars"], ["science fiction"], ["dark knight"]

This enables the model to distinguish between generic terms like "fiction" and specific genre markers like "science fiction".

5. Memory Optimization via Compressed Sparse Row (CSR) Format

The vectorizer produces a matrix of shape: $$\text{Matrix Dimensions: } 45,447 \text{ documents} \times 1,068,907 \text{ features}$$

The Dense Memory Problem: Storing $45,447 \times 1,068,907$ as standard 64-bit IEEE floating-point numbers would consume: $$45,447 \times 1,068,907 \times 8 \text{ bytes} \approx 388.64 \text{ Gigabytes of RAM}$$ This would instantly crash any standard server.

The Sparse Solution (CSR): Since any individual movie description contains at most a few dozen unique words, $99.99%$ of matrix entries are zero. The Compressed Sparse Row (CSR) data structure stores only non-zero values using three 1D arrays (data, indices, indptr): $$\text{Stored Elements: } 2,711,647 \quad (\approx 32.7 \text{ Megabytes})$$ This represents an over $10,000\times$ memory reduction, allowing the entire vector space to reside directly in RAM.

Phase 4: Similarity & Recommendation Ranking

Once the movies are represented as TF-IDF vectors, similar movies can be found by comparing their vectors.

1. Mathematical Comparison: Cosine Similarity vs. Euclidean Distance

Metric

Mathematical Formula

Behavior in High-Dimensional Text

Suitability

Cosine Similarity

$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}|_2 |\mathbf{v}|_2}$

Measures the angle between two vectors, completely independent of document length.

Optimal

Euclidean Distance

$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$

Measures the spatial distance between points. Severely distorted by document length.

Poor

Why Cosine Similarity Outperforms Euclidean Distance

Consider a short synopsis of 20 words for Movie A, and an expanded, comprehensive 200-word synopsis for Movie B that covers the exact same themes.

In Euclidean space, the coordinate magnitudes of Movie B will be substantially larger, placing the two films far apart in distance even though their narrative content is identical.

In Cosine space, because the angle $\theta$ between the vectors is evaluated, both films point along the same geometric ray: $$\cos(0^\circ) = 1.0 \quad \text{(Maximum Semantic Overlap)}$$

2. Optimization: Dot Product on $L_2$-Normalized Vectors

Because TfidfVectorizer outputs $L_2$-normalized vectors where $|\mathbf{u}|_2 = 1$ and $|\mathbf{v}|_2 = 1$, the standard Cosine formula simplifies to a matrix-vector dot product: $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{1 \times 1} = \mathbf{u} \cdot \mathbf{v}$$

3. $O(N)$ Real-Time Inference (Avoiding Full Gram Matrix Construction)

Pre-computing and persisting the full pairwise similarity matrix ($\mathbf{M} \times \mathbf{M}^T$) for 45,447 movies would require storing $45,447^2 \approx 2.06 \times 10^9$ values ($\sim 16.5 \text{ GB}$).

Instead, the system performs on-the-fly matrix-vector multiplication:

python

# 1. Retrieve query vector (qv) for target movie

qv = tfidf_matrix[idx]  # Shape: (1, 1068907)

# 2. Compute dot products across all 45,447 items via sparse linear algebra

scores = (tfidf_matrix @ qv.T).toarray().ravel()  # Shape: (45447,)

# 3. Descending rank retrieval (skipping self-match at index 0)

order = np.argsort(-scores)

This computation executes in under 15 milliseconds on a standard CPU.

Phase 5: FastAPI Backend

The inference engine is wrapped in an asynchronous REST service built with FastAPI (main.py).

1. Zero-Latency Warmup via lifespan

To prevent disk I/O bottlenecks during request handling, the application loads all model artifacts into memory once at application launch:

python

@asynccontextmanager

async def lifespan(app: FastAPI):

global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

with open("df.pkl", "rb") as f: df = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f: tfidf_matrix = pickle.load(f)

with open("indices.pkl", "rb") as f: indices_obj = pickle.load(f)

TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

yield

2. $O(1)$ Reverse-Index Lookups

Looking up a movie title by linear scanning of a 45,000-row DataFrame is an $O(N)$ operation. The backend maps all titles into an in-memory hash map (TITLE_TO_IDX), reducing index resolution to $O(1)$ time complexity: $$\text{Query Title} \xrightarrow{\text{Normalize & Hash}} \text{Integer Row Index } i$$

3. The Hybrid "Search Bundle" Pattern (/movie/search)

To solve catalog cold-starts and enhance pure content-based filtering, the backend implements a hybrid recommendation endpoint:

Local NLP Resolution: Queries the local TF-IDF model for top-$N$ semantic matches.

TMDB Real-Time Enrichment: Uses asynchronous HTTP clients (httpx.AsyncClient) to concurrently fetch:

Verified TMDB ID and canonical title.

High-resolution posters (w500) and backdrops.

Primary genre categorization.

Genre Discovery Fallback: Executes a secondary query against TMDB's /discover/movie?with_genres={id} to surface popular contemporary films matching the primary genre.

Phase 6: Streamlit Frontend

The user-facing layer is built with Streamlit (app.py), engineered as a single-page reactive web application with state persistence.

┌────────────────────────────────────────────────────────┐

│                      Streamlit UI                      │

│                                                        │

│  ┌──────────────────────────────────────────────────┐  │

│  │ 🔍 Search Input (Debounced Keyword Autocomplete) │  │

│  └──────────────────────────────────────────────────┘  │

│                                                        │

│  [ Home Feed Mode ]              [ Details Mode ]      │

│  - Trending                      - Poster & Backdrop   │

│  - Popular                       - Plot Synopsis       │

│  - Top-Rated                     - Similar Movies (NLP)│

│  - Upcoming                      - More Like This(TMDB)│

└────────────────────────────────────────────────────────┘

1. State Management & Browser Deep-Linking

Streamlit natively re-executes the entire Python script on every user interaction. To prevent session loss:

Page state is synchronized with browser URL parameters via st.query_params:

?view=home $\rightarrow$ Standard discovery grid.

?view=details&id=550 $\rightarrow$ Detailed view for TMDB Movie ID 550.

This enables browser bookmarking, back/forward navigation, and persistent state across page reloads.

2. Autocomplete Debounce & Word-Match Filtering

When users type in the search bar, the UI sends asynchronous requests to /tmdb/search. Results are filtered in real-time to match title substrings and presented in an interactive st.selectbox, eliminating typos and missing matches.

3. Responsive Poster Grids

The rendering engine calculates dynamic column subdivisions: $$\text{Rows} = \lceil \frac{\text{Total Cards}}{\text{Selected Columns}} \rceil$$ Each cell renders a movie card with high-resolution poster artwork, title truncation to prevent layout breaking, and an event listener button that triggers detail navigation.

Phase 7: User Activity & SQLite

The system records user interactions in a persistent SQLite database (user_history.db) managed via 

db.py

db.py.

Relational Schema Diagram

┌───────────────────────┐

│         users         │

├───────────────────────┤

│ user_id (UUID, PK)    │◄──────────┐

│ gender (TEXT)         │           │

│ country (TEXT)        │           │

│ created_at (DATETIME) │           │

└───────────────────────┘           │

                                │ (Foreign Key)

   ┌────────────────────────────┼────────────────────────────┐

   ▼                            ▼                            ▼

┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐

│    search_history    │   │    click_history     │   │    watch_history     │

├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤

│ id (INTEGER, PK)     │   │ id (INTEGER, PK)     │   │ id (INTEGER, PK)     │

│ user_id (TEXT, FK)   │   │ user_id (TEXT, FK)   │   │ user_id (TEXT, FK)   │

│ query (TEXT)         │   │ movie_id (INTEGER)   │   │ movie_id (INTEGER)   │

│ timestamp (DATETIME) │   │ title (TEXT)         │   │ title (TEXT)         │

└──────────────────────┘   │ timestamp (DATETIME) │   │ watch_duration_sec   │

                       └──────────────────────┘   │ timestamp (DATETIME) │

                                                  └──────────────────────┘

Telemetry Pipeline

Demographic Profiling: Upon first arrival, users are assigned a unique UUID4. Demographics (gender, country) are recorded in the users table.

Search Auditing (log_search): Captures query intent and search volume over time.

Implicit Feedback (log_click): Records item selection events when a user clicks "Open" on a movie card.

Engagement Quantification (log_watch): Tracks session watch duration in seconds. This provides explicit retention metrics for future Collaborative Filtering and Matrix Factorization algorithms.

Installation & Setup

1. Prerequisites

Python 3.10+

Free TMDB API Key from themoviedb.org

2. Environment Setup

bash

# Clone the repository

git clone https://github.com/your-username/movie-recommender-system.git

cd movie-recommender-system

# Create and activate virtual environment

python -m venv venv

# Windows:

venv\Scripts\activate

# macOS/Linux:

source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Download required NLTK tokenizers and lexical corpora

python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

3. Environment Configuration

Create a .env file in the root directory:

env

TMDB_API_KEY=your_actual_tmdb_api_key_here

4. Running the System

Start both the backend API and frontend application in separate terminals:

bash

# Terminal 1: Launch FastAPI Backend Server

uvicorn main --reload --port 8000

Interactive Swagger documentation will be available at: http://127.0.0.1:8000/docs

bash

# Terminal 2: Launch Streamlit Web UI

streamlit run app.py

Access the application interface at: http://localhost:8501

Technical Glossary

Bag-of-Words (BoW): An NLP representation that models text as an unordered collection of words, disregarding grammar and word order but maintaining frequency counts.

$TF\text{-}IDF$: Term Frequency-Inverse Document Frequency. A numerical statistic intended to reflect how important a word is to a document in a collection or corpus.

Lemmatization: The algorithmic process of determining a word's canonical base form (lemma) based on its intended meaning and morphological analysis.

Sparse Matrix (CSR): A matrix populated primarily with zeros, stored using compressed row pointers and column indexes to save memory and accelerate linear algebra computations.

Cosine Similarity: The cosine of the angle between two non-zero vectors in an inner product space, bounded between $[-1, 1]$ (or $[0, 1]$ for non-negative $TF\text{-}IDF$ space).

Cold-Start Problem: A well-known challenge in recommendation engines where the system cannot draw inferences for users or items for which it has not yet gathered sufficient information.

---

Key Improvements

Strictly Chronological: Flows logically through raw data $\rightarrow$ EDA $\rightarrow$ NLP cleaning $\rightarrow$ TF-IDF modeling $\rightarrow$ Cosine math $\rightarrow$ FastAPI backend $\rightarrow$ Streamlit frontend $\rightarrow$ SQLite tracking.

Deep Mathematical & Technical Rigor: Includes complete formulas for $TF$, $IDF$, $L_2$ normalization, Cosine Similarity, and CSR memory calculations.

Engineering Explanations: Explains why decisions were made (e.g., Lemmatization vs. Stemming, Cosine vs. Euclidean, on-the-fly dot products vs. full $O(N^2)$ precomputed similarity matrices, and CSR memory footprint savings).

Matches Your Exact Codebase: Accurately reflects movies_metadata.csv, recommender.ipynb, main.py, app.py, and db.py.
