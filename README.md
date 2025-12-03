# Movie Recommendation System

Find the perfect movie for your mood in seconds. Answer 4 quick questions, get 10 personalized recommendations.

## What It Does

Instead of endless scrolling, this system asks:
1. What's your plan tonight? (date, family, chill, friends)
2. Pick 1-2 genres (action, comedy, thriller, etc.)
3. Any specific themes? (keywords like "espionage", "heist")
4. What era? (recent, modern, classic, or any)

Then shows 10 movies with ratings, descriptions, and posters.

## Why It's Better

- **Context-aware**: Date night movies ≠ family night movies
- **No algorithm manipulation**: Not optimized for profit like Netflix
- **Quality filter**: Only shows movies worth your time (6.0+ rating minimum)
- **Real data**: 45,000 movies from IMDb + TMDb with full metadata

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
{"api_key": "a55b214aa396861a2625258556bbc6ee"}
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

## How It Works

**Data Pipeline**:
1. Clean IMDb data (45K movies, 500+ votes each)
2. Fetch TMDb metadata (posters, keywords, descriptions)
3. Extract country info for regional preferences
4. Train 4 ML models (collaborative filtering, content similarity, co-occurrence, keywords)

**Recommendation Engine**:
- **30% Collaborative Filtering**: "Users like you also liked..."
- **25% Content Similarity**: Genre + keyword matching
- **15% Co-occurrence**: Movies watched together
- **15% Quality Score**: Rating + vote reliability
- **15% Era Match**: Time period preference

Plus keyword boosts, region preference, diversity filtering.

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

## REST API

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

## Dataset

- 45,207 movies (1920-2025)
- 500+ votes minimum
- ~40K with TMDb enrichment
- 15+ genres, thousands of keywords

## Technical Details

See `general.info.md` for complete architecture, algorithms, and implementation details.


