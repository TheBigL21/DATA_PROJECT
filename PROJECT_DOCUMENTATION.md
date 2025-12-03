# Movie Recommendation System - Complete Documentation

A personalized movie recommendation system that helps you find the perfect movie based on your mood, preferences, and viewing context.

## Table of Contents
1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [Data Pipeline](#data-pipeline)
5. [Machine Learning Models](#machine-learning-models)
6. [Recommendation Engine](#recommendation-engine)
7. [Recent Updates](#recent-updates)

---

## System Overview

### What It Does

Instead of endless scrolling, this system asks 4 quick questions:
1. What's your plan tonight? (date, family, chill, friends)
2. Pick 1-2 genres (action, comedy, thriller, etc.)
3. Any specific themes? (keywords like "espionage", "heist")
4. What era? (recent, modern, classic, or any)

Then shows 10 personalized movies with ratings, descriptions, and posters.

### Key Features
- **Context-aware**: Date night movies ≠ family night movies
- **No algorithm manipulation**: Not optimized for profit like streaming services
- **Quality filter**: Automatic 6.0+ rating minimum, intelligent scoring
- **Genre-focused**: 25% of recommendation score based on explicit genre matching
- **Real data**: 45,000 movies from IMDb + TMDb with full metadata

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get Data

Download IMDb datasets:
```bash
mkdir imdb_raw && cd imdb_raw
wget https://datasets.imdbws.com/title.basics.tsv.gz
wget https://datasets.imdbws.com/title.ratings.tsv.gz
wget https://datasets.imdbws.com/title.principals.tsv.gz
wget https://datasets.imdbws.com/title.akas.tsv.gz
wget https://datasets.imdbws.com/name.basics.tsv.gz
cd ..
```

Get TMDb API key:
- Sign up at https://www.themoviedb.org/
- Create `config/tmdb_config.json`:
```json
{"api_key": "your_api_key_here"}
```

### 3. Run Pipeline
```bash
python run_pipeline.py
```
Takes 2-4 hours (TMDb API rate-limited). Progress saved automatically.

### 4. Get Recommendations
```bash
python movie_finder.py
```

---

## How It Works

### Data Sources

**IMDb Dataset** (Public)
- Movie titles, years, genres, runtime
- Ratings and vote counts
- Cast and crew information
- Coverage: 45,207 quality movies (≥500 votes)

**TMDb API**
- Movie posters and images
- Plot descriptions
- Keywords (thematic tags like "espionage", "heist")
- Budget and revenue data
- 99.3% coverage of our dataset

### Recommendation Algorithm

**Hybrid Scoring System** - 6 weighted components:

1. **Genre Matching (25%)** ← PRIMARY USER INTENT
   - Movies matching both selected genres: 1.0 score
   - Movies matching one genre: 0.5 score
   - Ensures recommendations respect user's explicit selection

2. **Collaborative Filtering (25%)**
   - "Users like you also liked..."
   - Matrix factorization (SVD) with 50 latent factors
   - Based on synthetic user behavior patterns

3. **Quality Score (15%)**
   - Automatic 6.0+ minimum filter
   - Linear scoring from 6.0-10.0
   - Vote confidence multiplier (more votes = higher confidence)

4. **Co-occurrence Graph (15%)**
   - Movies frequently watched together
   - "People who liked X also liked Y"
   - Graph-based recommendations

5. **Session Similarity (10%)**
   - Genre overlap with current session
   - Year similarity to recently viewed movies
   - Adapts as user swipes

6. **Era Favorability (5%)**
   - Soft time period preference
   - Movies in range get 1.0, outside decay 2%/year
   - No hard cutoffs - great old movies still appear

7. **Keyword Matching (5%)**
   - Exact keyword match in movie metadata
   - Partial match in descriptions
   - User selects 0-3 refinement keywords

---

## Data Pipeline

### Stage 1: IMDb Data Cleaning
**File**: `src/data_processing/data_clean.py`

Filters:
- Only movies (not TV shows, shorts)
- Not adult content
- Runtime 60-400 minutes
- ≥500 votes (quality control)
- Valid year (1920-2025)
- Has genres and ratings

Output: ~45,000 quality movies from millions

### Stage 2: Country Extraction
**File**: `src/data_processing/extract_country_fast.py`

- Maps each movie to production country
- Used for regional preference weighting

### Stage 3: TMDb Enrichment
**File**: `src/data_processing/fetch_tmdb_data.py`

Fetches for each movie:
- Plot descriptions
- High-quality posters
- Keywords (thematic tags)
- Additional ratings
- Budget/revenue data

Rate-limited to 10 req/sec. Takes 2-4 hours for full dataset.

### Stage 4: Data Transformation
**File**: `src/data_processing/transform_imdb_data.py`

- Joins all tables into single denormalized schema
- Aggregates genres and keywords into lists
- Creates composite rating: (IMDb + TMDb) / 2
- Assigns integer movie_id for ML efficiency

### Stage 5: Synthetic Interactions
**File**: `src/data_processing/generate_synthetic_interactions.py`

- Generates 1000 synthetic users with realistic patterns
- Simulates ratings based on genre preferences, quality bias, regional preferences
- Needed for collaborative filtering (cold-start problem)

---

## Machine Learning Models

### Model 1: Collaborative Filtering
**Algorithm**: Singular Value Decomposition (SVD)

- Learns user and movie latent factors (50 dimensions)
- Predicts: "How much would this user like this movie?"
- Trained on synthetic interaction data
- Captures: "Users like you also liked..."

### Model 2: Co-occurrence Graph
**Algorithm**: Item-based collaborative filtering

- Builds graph of movies watched together
- Edge weight = co-occurrence frequency
- Captures: "People who watched X also watched Y"
- Sparse adjacency matrix (45K × 45K)

### Model 3: Content Similarity
**Algorithm**: TF-IDF + Cosine Similarity

- Creates feature vectors from genres, keywords, directors
- Computes pairwise similarity (45K × 45K)
- Identifies thematically similar movies

### Model 4: Keyword Database
**Algorithm**: Inverted index with frequency ranking

- Maps keywords to movie lists
- Enables fast lookup and contextual suggestions
- Genre-specific keyword recommendations

---

## Recommendation Engine

**File**: `src/recommendation/smart_engine.py`

### Process Flow

**Stage 1: Candidate Generation** (~500-1000 movies)
- Hard genre filter (must match selected genres)
- CF top predictions (filtered by genre)
- Graph neighbors of positive session movies
- Popular movies in genre (cold start fallback)

**Stage 2: Composite Scoring**
```python
score = 0.25 × genre_score +      # NEW - explicit genre matching
        0.25 × cf_score +          # Collaborative filtering
        0.15 × quality_score +     # Linear quality 6.0-10.0
        0.15 × graph_score +       # Co-occurrence
        0.10 × session_score +     # Within-session similarity
        0.05 × era_score +         # Time period preference
        0.05 × keyword_score       # Keyword matching
```

**Stage 3: Ranking & Return**
- Sort by composite score
- Return top K (typically 10)
- Exclude already shown movies

---

## Recent Updates

### Genre Weighting Fix (Dec 2025)

**Problem**: When users selected "Action + Thriller," only 40% of recommendations matched both genres. CF model was overriding user intent.

**Solution**: Added explicit genre scoring component (25% weight)
- Movies matching both genres: 1.0 score (full 25%)
- Movies matching one genre: 0.5 score (12.5%)
- Rebalanced other weights to accommodate

**Result**: 100% genre match rate (up from 40%), maintaining 7.9/10 avg quality

### Quality Scoring Automation (Dec 2025)

**Removed**: User quality preference question

**Added**: Automatic quality handling
- Hard 6.0+ filter for all candidates
- Linear scoring from 6.0-10.0
- Vote confidence multiplier (log scale)
- Simpler user flow, consistent quality standards

### Keyword System (Dec 2025)

**Added**: Genre-contextual keyword suggestions
- System suggests 6 relevant keywords based on selected genres
- Uses TF-IDF analysis on movie metadata
- User can select 0-3 to refine search
- Excludes generic/metadata keywords

---

## User Flow

1. **Evening Type**: Chill / Date / Family / Friends
2. **Genres**: Pick 1-2 (action, comedy, thriller, etc.)
3. **Keywords**: Optional 0-3 refinements (espionage, heist, etc.)
4. **Era**: Recent / Modern / Timeless / Any
5. **Results**: Top 10 personalized recommendations

**No quality question** - handled automatically!

---

## Project Structure

```
src/
  data_processing/     # IMDb cleaning, TMDb fetching, transformations
  models/              # CF, content similarity, co-occurrence
  recommendation/      # Main engine, keyword system, region weights
  auth/                # User authentication

movie_finder.py        # Interactive CLI
api_smart.py           # REST API
run_pipeline.py        # Data pipeline orchestrator
```

---

## API Usage

```bash
python api_smart.py
```

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "evening_type": "date_night",
    "selected_genres": ["action", "thriller"],
    "selected_keywords": ["espionage"],
    "age_preference": "modern",
    "user_region": "US"
  }'
```

---

## Dataset Statistics

- **45,207 movies** (1920-2025)
- **500+ votes minimum** per movie
- **~40K with TMDb enrichment** (posters, keywords, descriptions)
- **15+ genres**, thousands of keywords
- **Mean rating**: 6.2/10
- **Quality range**: All movies pass 6.0+ threshold

---

## Testing

```bash
python test_system.py
```

Tests comprehensive scenarios across different genres and evening types.

Expected: 100% genre match rate, 7.9+/10 average quality
