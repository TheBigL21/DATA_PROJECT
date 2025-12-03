# Movie Recommendation System - Complete Documentation

**Last Updated**: December 2, 2025
**Status**: Production Ready
**Version**: Smart System with Normal Law Quality Scoring

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Smart Recommendation Engine](#smart-recommendation-engine)
5. [Normal Law Quality Scoring](#normal-law-quality-scoring)
6. [Data Schema](#data-schema)
7. [API Endpoints](#api-endpoints)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Future Enhancements](#future-enhancements)

---

## System Overview

Mood-driven movie recommendation system combining collaborative filtering, co-occurrence graphs, and intelligent quality scoring to minimize swipes before finding the perfect movie.

### Key Features
- **Smart Quality Scoring**: Normal distribution favoring excellent movies (8.0+)
- **Era Favorability**: Soft time preferences without hard filtering
- **Keyword System**: Contextual keyword extraction and matching
- **Three-Level Swipe Feedback**: Left (reject), Right (interested), Up (select)
- **Session-Aware Learning**: Adapts recommendations as user swipes

### Tech Stack
- Python 3.8+
- Pandas, NumPy, SciPy
- Flask API
- Collaborative Filtering (ALS)
- Co-occurrence Graph
- TF-IDF for keywords

---

## Quick Start

### 1. Download IMDb Data
```bash
mkdir imdb_raw && cd imdb_raw
wget https://datasets.imdbws.com/title.basics.tsv.gz
wget https://datasets.imdbws.com/title.ratings.tsv.gz
wget https://datasets.imdbws.com/title.principals.tsv.gz
wget https://datasets.imdbws.com/title.crew.tsv.gz
wget https://datasets.imdbws.com/name.basics.tsv.gz
cd ..
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py \
  --imdb-dir ./imdb_raw \
  --output-dir ./output \
  --num-users 1000 \
  --sessions-per-user 10
```

### 3. Start API Server
```bash
python3 api_smart.py
# Server runs on http://localhost:5000
```

### 4. Test System
```bash
python3 test_smart_recommendations.py
```

---

## Architecture

### Three-Layer Design

#### Layer 1: Data Processing
- **Input**: IMDb TSV dumps + TMDB API data
- **Output**: Clean parquet files (movies, interactions, people)
- **Schema**: Optimized for ML (integer movie_ids, sparse matrices)

#### Layer 2: Machine Learning Models

**Collaborative Filtering (ALS)**
- Matrix factorization learning user preferences
- User and movie latent vectors (50 dimensions)
- Trained on three-level swipe feedback (0.0, 0.3, 1.0)

**Co-occurrence Graph**
- Captures "users who liked X also liked Y" patterns
- Sparse adjacency matrix (movie x movie)
- Edge weights from session co-occurrence

**Content Similarity (TF-IDF)**
- Text similarity based on descriptions
- Keyword extraction per genre
- Semantic matching for refinement

#### Layer 3: Recommendation Engine

**Smart Engine Components** (6 scoring factors):
1. **Collaborative Filtering (30%)** - User preferences
2. **Co-occurrence Graph (20%)** - Movie relationships
3. **Session Similarity (15%)** - Within-session context
4. **Quality Score (15%)** - Normal law quality (see below)
5. **Era Favorability (10%)** - Soft time preference
6. **Keyword Matching (10%)** - Contextual relevance

---

## Smart Recommendation Engine

### File Location
`src/recommendation/smart_engine.py` (490 lines)

### User Flow

**Step 1: Evening Type**
- Chill Evening (solo, experimental)
- Date Night (impressive movies)
- Family Night (reliable entertainment)
- Friends Night (social viewing)

**Step 2: Genres** (1-2 selections)
- Action, Comedy, Drama, Horror, Romance, Sci-Fi, etc.

**Step 3: Era**
- Fresh Picks (2020-2025)
- Modern Classics (2015-2025)
- Timeless Favorites (2000-2025)
- Old-School Gems (1900-2025)

**Step 4: Keywords** (optional, 0-2)
- System suggests 8 contextual keywords based on genres
- Example: Action + Thriller → espionage, heist, conspiracy

### Recommendation Process

**Stage 1: Candidate Generation (~500-1000 movies)**
- Hard genre filter (must match selected genres)
- CF top predictions (filtered by genre)
- Graph neighbors of session positive movies
- Popular movies in genre (cold start)

**Stage 2: Composite Scoring**
```python
score = 0.30 * cf_score +
        0.20 * graph_score +
        0.15 * session_score +
        0.15 * quality_score +  # ← Normal law applied here
        0.10 * era_score +
        0.10 * keyword_score
```

**Stage 3: Ranking & Return**
- Sort by composite score
- Return top K (default 20)
- Exclude already shown movies

---

## Normal Law Quality Scoring

### Implementation Details

**Feature Name**: `normal_law_try`
**Date Implemented**: December 2, 2025
**Location**: `src/recommendation/smart_engine.py:337-397`

### Motivation

User requested avoiding bad movies without hard rating filters. Solution: Use normal distribution to naturally prioritize excellent movies while maintaining mathematical smoothness.

### Algorithm

**Parameters**:
- Mean (μ) = 8.0
- Standard Deviation (σ) = 1.5

**Formula**:
```python
z_score = (rating - 8.0) / 1.5
gaussian = exp(-0.5 * z_score²)

if rating >= 7.5:
    # Boost excellent movies
    base_quality = min(1.0, gaussian * (1 + (rating - 7.5) * 0.15))
else:
    # Natural decay for lower ratings
    base_quality = gaussian

# Apply confidence and context
quality_score = base_quality * confidence * evening_modifier
```

**Note**: This is NOT a true symmetric normal distribution. It has an asymmetric boost for ratings ≥7.5 to favor classics.

### Score Distribution

| Rating | Base Quality | Interpretation |
|--------|-------------|----------------|
| 9.5    | 0.788       | Masterpieces |
| 9.0    | 0.981       | Masterpieces |
| 8.5    | 1.000       | Classics/Excellent (peak) |
| 8.0    | 1.000       | Classics/Excellent (peak) |
| 7.5    | 0.946       | Very good |
| 7.0    | 0.801       | Good |
| 6.5    | 0.607       | Decent |
| 6.0    | 0.411       | Mediocre |
| 5.5    | 0.249       | Rarely shown |
| 5.0    | 0.135       | Almost never shown |
| <4.0   | <0.030      | Virtually never shown |

### Confidence Multiplier (Vote Count)
```python
if num_votes >= 100,000: confidence = 1.0
elif num_votes >= 10,000: confidence = 0.9
elif num_votes >= 1,000:  confidence = 0.7
else:                     confidence = 0.5
```

### Evening Type Modifier
```python
modifiers = {
    'date_night': 1.2,      # Need impressive movies
    'family_night': 1.1,    # Want reliable entertainment
    'friends_night': 1.0,   # Social focus
    'chill_evening': 0.9    # Experimental mood
}
```

### Impact

**Practical Effects**:
- Movies rated 8.0-8.5 get maximum quality scores (peak of distribution)
- Movies rated 7.0-7.5 remain competitive (still appear frequently)
- Movies rated <6.0 get very low quality scores (rarely surface)
- No hard cutoff: even bad movies can appear if CF/keywords predict user interest

**Weight in Overall Score**: 15%
This means quality is important but not dominant—personalization (CF) and relationships (graph) still matter most.

### Tuning Parameters

If you want to adjust the curve:
- **mean**: Shift center (7.5 = more forgiving, 8.5 = stricter)
- **std**: Change steepness (1.2 = steeper, 2.0 = gentler)
- **boost_factor**: Adjust high-rating boost (currently 0.15)
- **w_quality**: Change overall weight (currently 15%, could be 20-30%)

---

## Data Schema

### movies.parquet (45,207 movies)

**Core Fields**:
- `movie_id` (int): 0-indexed internal ID
- `tconst` (str): IMDb ID (e.g., tt0111161)
- `title` (str): Movie title
- `year` (int): Release year
- `runtime` (int): Minutes
- `genres` (list): Genre tags

**Ratings**:
- `avg_rating` (float): IMDb rating (1-10)
- `tmdb_rating` (float): TMDB rating (1-10)
- `composite_rating` (float): 60% IMDb + 40% TMDB
- `num_votes` (int): IMDb vote count (all ≥1000)

**People**:
- `director` (str): Director name
- `actors` (list): Top 5 actors
- `country` (str): Production country

**TMDB Data** (99.3% coverage):
- `tmdb_id` (float): TMDB ID
- `description` (str): Plot summary
- `poster_url` (str): Poster image URL
- `backdrop_url` (str): Backdrop image URL
- `keywords` (np.ndarray): TMDB keyword tags
- `budget` (float): Production budget
- `revenue` (float): Box office revenue

---

## API Endpoints

### Base URL
`http://localhost:5000`

### 1. Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "ok",
  "num_movies": 45207,
  "has_keywords": true
}
```

### 2. Get Questionnaire Options
```bash
GET /api/questionnaire/options
```
Returns all available evening types, genres, and eras.

### 3. Get Keyword Suggestions
```bash
POST /api/questionnaire/keywords
Content-Type: application/json

{
  "genres": ["action", "thriller"]
}
```
Response:
```json
{
  "keywords": ["espionage", "conspiracy", "heist", "undercover", ...]
}
```

### 4. Get Recommendations
```bash
POST /api/recommend
Content-Type: application/json

{
  "user_id": 0,
  "evening_type": "date_night",
  "genres": ["action", "thriller"],
  "era": "modern",
  "keywords": ["espionage"],
  "session_history": [
    {"movie_id": 100, "action": "left"},
    {"movie_id": 250, "action": "right"}
  ],
  "top_k": 20
}
```
Response:
```json
{
  "recommendations": [
    {
      "movie_id": 12345,
      "title": "Mission: Impossible - Fallout",
      "year": 2018,
      "genres": ["action", "thriller"],
      "avg_rating": 7.7,
      "tmdb_rating": 7.4,
      "combined_rating": 7.58,
      "num_votes": 345000,
      "description": "Ethan Hunt and the IMF team...",
      "poster_url": "https://...",
      "keywords": ["espionage", "cia"]
    },
    ...
  ]
}
```

### 5. Get Movie Details
```bash
GET /api/movie/<movie_id>
```
Returns detailed information for a specific movie.

### 6. Record Feedback
```bash
POST /api/feedback
Content-Type: application/json

{
  "user_id": 0,
  "movie_id": 12345,
  "action": "up"
}
```

---

## Project Structure

```
DATA_PROJECT/
├── data/
│   └── data_clean.py                      # IMDb data cleaning
│
├── src/
│   ├── data_processing/
│   │   ├── transform_imdb_data.py         # Schema transformation
│   │   ├── fetch_tmdb_data.py             # TMDB API integration
│   │   └── generate_synthetic_interactions.py
│   │
│   ├── models/
│   │   ├── collaborative_filtering.py     # ALS model
│   │   ├── cooccurrence_graph.py          # Movie-movie graph
│   │   └── content_similarity.py          # TF-IDF
│   │
│   ├── recommendation/
│   │   ├── recommendation_engine.py       # Base engine
│   │   ├── smart_engine.py                # ★ Smart engine with normal law
│   │   └── keyword_analyzer.py            # Keyword system
│   │
│   └── analytics/
│       └── genre_tracker.py
│
├── output/
│   ├── processed/
│   │   └── movies.parquet                 # 45,207 movies (19MB)
│   │
│   └── models/
│       ├── user_factors.npy               # CF model
│       ├── movie_factors.npy
│       ├── adjacency_matrix.npz           # Graph
│       ├── content_similarity.pkl
│       └── keyword_database.pkl
│
├── config/
│   └── genre_allocation.json
│
├── api.py                                  # Legacy API
├── api_smart.py                            # ★ Smart API
├── movie_finder.py                         # CLI interface
├── test_smart_recommendations.py           # ★ Test suite
├── run_pipeline.py                         # Pipeline orchestrator
│
└── PROJECT_DOCUMENTATION.md                # ★ This file
```

---

## Testing

### Run Test Suite
```bash
python3 test_smart_recommendations.py
```

### Expected Output
```
✓ Option C quality scoring test passed
✓ Era favorability working correctly (not filtering)
✓ Complete flow test passed
✓ ALL TESTS PASSED!
```

### Manual Testing
```bash
# Start server
python3 api_smart.py

# In another terminal
curl http://localhost:5000/health
curl http://localhost:5000/api/questionnaire/options
```

---

## Future Enhancements

### Priority: High
1. **Optimize Keyword Database**
   - Replace TF-IDF with frequency counting
   - Reduce build time from 40min to 2min

2. **Add Real User Data Collection**
   - Log all recommendations and swipe actions
   - Periodically retrain CF model
   - Update graph edges with new sessions

### Priority: Medium
3. **Add Description Similarity Component**
   - Use TF-IDF on plot descriptions
   - Add as 7th scoring component (5% weight)

4. **A/B Testing Framework**
   - Test different weight configurations
   - Compare normal law vs exponential quality scoring
   - Measure: swipes to success, session completion rate

### Priority: Low
5. **Streaming Availability Integration**
   - Integrate with JustWatch API
   - Filter by Netflix/Prime/Disney+

6. **Deep Learning Exploration**
   - Neural collaborative filtering
   - Transformer-based embeddings for descriptions

---

## Performance Targets

### Offline Metrics
- Hit Rate@10: >60%
- Mean swipes to success: <10

### Online Metrics
- Session completion rate: >70%
- Right-swipe rate: 30-50%
- API response time: <100ms

---

## Dependencies

```bash
pip install pandas numpy scipy flask flask-cors
```

**Python Version**: 3.8+

---

## Key Design Decisions

### 1. Why Normal Law for Quality?
- Naturally favors excellent movies (8.0+) without hard cutoffs
- Mathematical smoothness (no artificial boundaries)
- Context-aware (evening type modifies scoring)
- Transparent (users can understand the logic)

### 2. Why Era Favorability (Not Filtering)?
- Hard filters exclude great movies just outside range
- Smooth decay preserves flexibility
- Old movies can still rank high if other factors strong
- Better user experience (fewer disappointments)

### 3. Why Three-Level Swipe Feedback?
- Binary like/dislike too coarse
- "Right" (0.3) captures "interesting but not quite"
- Guides exploration without commitment
- More training signal for ML models

### 4. Why Simplified User Flow?
- Old 6-question flow caused user fatigue
- Quality now implicit (handled by normal law scoring)
- Era now soft preference (not hard filter)
- Faster, more flexible experience

---

## Troubleshooting

### Issue: API Won't Start
**Check dependencies**:
```bash
pip3 install flask flask-cors pandas numpy scipy
```

**Check models exist**:
```bash
ls output/models/
# Should see: user_factors.npy, movie_factors.npy, adjacency_matrix.npz
```

### Issue: Recommendations All Recent Movies
**This is expected** with "fresh" era (2020-2025). Try:
- Use "old_school" era (1900-2025) for full variety
- Check that CF model is loaded correctly
- Verify genre filtering isn't too restrictive

### Issue: Quality Scores Too Strict
**Adjust normal law parameters**:
- Decrease mean (8.0 → 7.5) for more forgiving scoring
- Increase std (1.5 → 2.0) for gentler curve
- Edit `src/recommendation/smart_engine.py:362-365`

---

## Database Statistics

- **Total Movies**: 45,207
- **TMDB Coverage**: 99.3% (44,895 movies)
- **All Movies**: ≥1,000 votes
- **Mean Rating**: 6.2/10
- **Median Rating**: 6.4/10
- **75th Percentile**: 7.0/10

With normal law scoring:
- Top 25% (7.0+) get quality scores ≥0.80
- Top 10% (7.5+) get quality scores ≥0.95
- Bottom 50% (<6.4) get quality scores <0.50

---


