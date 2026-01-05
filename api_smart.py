"""
SMART RECOMMENDATION API SERVER

REST API for smart movie recommendation system with Option C scoring.

New Endpoints:
1. GET /api/quest ionnaire/options - Get all available options
2. POST /api/questionnaire/keywords - Get contextual keyword suggestions
3. POST /api/recommend - Generate recommendations (updated schema)
4. GET /api/movie/<movie_id> - Get detailed movie information

Usage:
  python3 api_smart.py

Then access at http://localhost:5000
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import logging
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


from recommendation.smart_engine import load_smart_system  # type: ignore
from recommendation.time_period_filter import TimePeriodFilter  # type: ignore
from recommendation.source_material_filter import SourceMaterialFilter  # type: ignore
from recommendation.keyword_recommender import KeywordRecommender  # type: ignore  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pandas to suppress chained assignment warnings
pd.options.mode.chained_assignment = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load recommendation system
# Use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "output" / "models"
MOVIES_PATH = SCRIPT_DIR / "output" / "processed" / "movies.parquet"
KEYWORD_DB_PATH = SCRIPT_DIR / "output" / "models" / "keyword_database.pkl"
FEEDBACK_DB_PATH = SCRIPT_DIR / "output" / "feedback.db"

logger.info("Loading smart recommendation system...")
try:
    engine = load_smart_system(MODELS_DIR, MOVIES_PATH, KEYWORD_DB_PATH, FEEDBACK_DB_PATH)
    
    # CRITICAL FILTER: Ensure only movies with >=1000 votes are used
    # This ensures consistency even if the parquet file wasn't properly filtered
    initial_count = len(engine.movies)
    if 'num_votes' in engine.movies.columns:
        engine.movies = engine.movies[engine.movies['num_votes'] >= 1000].copy()
        filtered_count = initial_count - len(engine.movies)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count:,} movies with <1000 votes")
            logger.info(f"Using {len(engine.movies):,} quality movies (>=1000 votes)")
        else:
            logger.info(f"All {len(engine.movies):,} movies have >=1000 votes ✓")
    else:
        logger.warning("'num_votes' column not found - cannot filter movies")
    
    logger.info("✓ Smart system loaded successfully")
    logger.info(f"  - {len(engine.movies):,} movies")
    logger.info(f"  - Keyword analyzer: {'✓' if engine.keyword_analyzer else '✗'}")
    logger.info(f"  - Content Similarity: {'✓' if engine.content_similarity else '✗'}")
    logger.info(f"  - Feedback Learner: {'✓' if engine.feedback_learner else '✗'}")
    
    # Initialize KeywordRecommender (same as interactive_movie_finder.py)
    # Wrap in try-except to handle gracefully if it fails
    try:
        logger.info("Initializing keyword recommender...")
        keyword_recommender = KeywordRecommender(str(MOVIES_PATH))
        logger.info("✓ Keyword recommender initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize KeywordRecommender: {e}")
        logger.warning("Falling back to engine.keyword_analyzer for keyword suggestions")
        keyword_recommender = None
except Exception as e:
    logger.error(f"Failed to load system: {e}")
    raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'num_movies': len(engine.movies),
        'has_keywords': engine.keyword_analyzer is not None
    }), 200


@app.route('/api/questionnaire/options', methods=['GET'])
def get_questionnaire_options():
    """
    Get all available questionnaire options.

    Response:
    {
        "evening_types": [...],
        "genres": [...],
        "eras": [...]
    }
    """
    try:
        # Extract unique genres from dataset
        all_genres = set()
        for genres in engine.movies['genres']:
            if isinstance(genres, (list, tuple)):
                all_genres.update(genres)

        return jsonify({
            'evening_types': [
                {
                    'id': 'chill_evening',
                    'label': 'Chill Evening by myself',
                    'description': 'Relaxed solo viewing, open to experimenting'
                },
                {
                    'id': 'date_night',
                    'label': 'Date Night',
                    'description': 'Impressive movies for a special evening'
                },
                {
                    'id': 'family_night',
                    'label': 'Family Night',
                    'description': 'Safe, reliable entertainment for everyone'
                },
                {
                    'id': 'friends_night',
                    'label': 'Friends Night',
                    'description': 'Fun social viewing experience'
                }
            ],
            'genres': sorted(list(all_genres)),
            'eras': TimePeriodFilter.get_era_options()
        }), 200

    except Exception as e:
        logger.error(f"Error in get_questionnaire_options: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questionnaire/genres', methods=['GET'])
def get_genre_config():
    """
    Get core and extended genres for a given evening type.
    
    Query params:
        evening_type: Frontend ID ('chill_evening', 'date_night', etc.)
    
    Response:
    {
        "core": ["thriller", "drama", ...],
        "extended": ["crime", "biography", ...]
    }
    """
    try:
        evening_type = request.args.get('evening_type')
        if not evening_type:
            return jsonify({'error': 'Missing required parameter: evening_type'}), 400
        
        # Map frontend IDs to backend strings
        evening_type_map = {
            'chill_evening': 'Chill Evening by myself',
            'date_night': 'Date night',
            'family_night': 'Family night',
            'friends_night': 'Friends night'
        }
        
        backend_type = evening_type_map.get(evening_type, evening_type)
        
        # Load genre_allocation.json - USE ABSOLUTE PATH
        config_path = SCRIPT_DIR / 'config' / 'genre_allocation.json'
        if not config_path.exists():
            return jsonify({'error': f'Genre configuration file not found at {config_path}'}), 500
        
        with open(config_path, 'r') as f:
            genre_config = json.load(f)
        
        if backend_type not in genre_config:
            return jsonify({'error': f'Invalid evening_type: {evening_type}'}), 400
        
        return jsonify(genre_config[backend_type]), 200
        
    except Exception as e:
        logger.error(f"Error in get_genre_config: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/questionnaire/keywords', methods=['POST'])
def generate_keywords():
    """
    Generate contextual keyword suggestions based on selections.

    Request:
    {
        "genres": ["action", "thriller"]
    }

    Response:
    {
        "keywords": ["espionage", "conspiracy", "heist", ...]
    }
    """
    try:
        data = request.get_json()

        # Validate
        if 'genres' not in data:
            return jsonify({'error': 'Missing required field: genres'}), 400

        genres = data['genres']
        if not isinstance(genres, list) or len(genres) == 0:
            return jsonify({'error': 'genres must be a non-empty list'}), 400

        # Generate keywords using KeywordRecommender if available, else fallback to engine
        if keyword_recommender:
            # Use KeywordRecommender (same as interactive_movie_finder.py)
            genres_lower = [g.lower().strip() for g in genres]
            keywords = keyword_recommender.get_keywords_for_genres(genres_lower, num_keywords=8)
        elif engine.keyword_analyzer:
            keywords = engine.suggest_keywords(genres, num_keywords=8)
        else:
            keywords = []

        return jsonify({'keywords': keywords}), 200

    except Exception as e:
        logger.error(f"Error in generate_keywords: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questionnaire/source-material', methods=['POST'])
def get_relevant_source_material():
    """
    Get relevant source material option for given genres.
    
    Request:
    {
        "genres": ["action", "thriller"]
    }
    
    Response:
    {
        "source_material": "book",
        "label": "Based on Book/Novel",
        "description": "Movies adapted from books or novels"
    }
    """
    try:
        data = request.get_json()
        genres = data.get('genres', [])
        
        if not isinstance(genres, list):
            return jsonify({'error': 'genres must be a list'}), 400
        
        # Same logic as get_relevant_source_material() in interactive_movie_finder.py
        genre_source_map = {
            'fantasy': 'book',
            'sci-fi': 'book',
            'romance': 'book',
            'drama': 'true_story',
            'biography': 'true_story',
            'history': 'true_story',
            'war': 'true_story',
            'action': 'comic',
            'adventure': 'book',
            'crime': 'true_story',
            'thriller': 'book',
            'mystery': 'book',
            'horror': 'book',
            'musical': 'play_musical',
            'comedy': 'book'
        }
        
        source_votes = {}
        for genre in genres:
            genre_lower = genre.lower()
            if genre_lower in genre_source_map:
                source = genre_source_map[genre_lower]
                source_votes[source] = source_votes.get(source, 0) + 1
        
        if source_votes:
            most_relevant = max(source_votes.items(), key=lambda x: x[1])[0]
        else:
            most_relevant = 'book'  # Default fallback
        
        source_info = SourceMaterialFilter.SOURCE_KEYWORDS[most_relevant]
        
        return jsonify({
            'source_material': most_relevant,
            'label': source_info['label'],
            'description': source_info['description']
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_relevant_source_material: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Generate movie recommendations with smart scoring.

    Request:
    {
        "user_id": int,
        "evening_type": str,  // 'date_night', 'family_night', 'friends_night', 'chill_evening'
        "genres": [str, str],  // 1-2 genres
        "era": str,  // 'new_era', 'millennium', 'old_school', 'golden_era', 'any'
        "keywords": [str, str],  // 0-3 keywords/themes (optional)
        "source_material": str,  // Optional: 'book', 'true_story', 'comic', 'play_musical', 'original', 'any'
        "session_history": [
            {"movie_id": int, "action": "yes|no|final"},
            ...
        ],
        "top_k": int (optional, default 20)
    }

    Response:
    {
        "recommendations": [
            {
                "movie_id": int,
                "title": str,
                "year": int,
                "genres": [str],
                "avg_rating": float,
                "tmdb_rating": float,
                "combined_rating": float,
                "num_votes": int,
                "director": str,
                "actors": [str],
                "description": str,
                "poster_url": str,
                "keywords": [str]
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['user_id', 'evening_type', 'genres', 'era']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract parameters
        user_id = data['user_id']
        evening_type = data['evening_type']
        genres = data['genres']
        era = data['era']
        keywords = data.get('keywords', [])
        source_material = data.get('source_material', 'any')
        session_history = data.get('session_history', [])
        top_k = data.get('top_k', 20)

        # Validate types
        if not isinstance(genres, list) or len(genres) == 0 or len(genres) > 2:
            return jsonify({'error': 'genres must be a list of 1-2 items'}), 400

        if keywords and not isinstance(keywords, list):
            return jsonify({'error': 'keywords must be a list'}), 400

        # Generate recommendations
        logger.info(f"Generating recommendations for user {user_id}")
        logger.info(f"  Evening: {evening_type}, Genres: {genres}, Era: {era}, Keywords: {keywords}, Source Material: {source_material}")

        # Extract session_id from request or generate one
        import time
        session_id = data.get('session_id', f'session_{user_id}_{int(time.time())}')
        
        movie_ids = engine.recommend(
            user_id=user_id,
            genres=genres,
            era=era,
            source_material=source_material,
            themes=keywords,  # API uses 'keywords' but engine expects 'themes'
            session_history=session_history,
            session_id=session_id,  # Pass session_id for cross-session learning
            top_k=top_k
        )

        # Format response
        recommendations = []
        for movie_id in movie_ids:
            movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0]

            # Calculate combined rating
            imdb_rating = movie['avg_rating']
            tmdb_rating = movie.get('tmdb_rating')
            if tmdb_rating and not pd.isna(tmdb_rating):
                combined_rating = (imdb_rating + tmdb_rating) / 2.0
            else:
                combined_rating = imdb_rating

            rec = {
                'movie_id': int(movie_id),
                'title': movie['title'],
                'year': int(movie['year']),
                'runtime': int(movie['runtime']),
                'genres': list(movie['genres']) if isinstance(movie['genres'], (list, tuple)) else [],
                'avg_rating': float(imdb_rating),
                'tmdb_rating': float(tmdb_rating) if tmdb_rating and not pd.isna(tmdb_rating) else None,
                'combined_rating': float(combined_rating),
                'num_votes': int(movie['num_votes']),
                'director': movie['director'],
                'actors': list(movie['actors']) if isinstance(movie['actors'], (list, tuple)) else [],
                'description': movie.get('description') if pd.notna(movie.get('description')) else None,
                'poster_url': movie.get('poster_url') if pd.notna(movie.get('poster_url')) else None,
                'keywords': list(movie.get('keywords', [])) if movie.get('keywords') is not None else []
            }
            recommendations.append(rec)

        logger.info(f"✓ Generated {len(recommendations)} recommendations")

        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        logger.error(f"Error in recommend: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """
    Get detailed information about a specific movie.

    Response:
    {
        "movie_id": int,
        "title": str,
        ...
        (same fields as recommend endpoint)
    }
    """
    try:
        movie = engine.movies[engine.movies['movie_id'] == movie_id]

        if len(movie) == 0:
            return jsonify({'error': 'Movie not found'}), 404

        movie = movie.iloc[0]

        # Calculate combined rating
        imdb_rating = movie['avg_rating']
        tmdb_rating = movie.get('tmdb_rating')
        if tmdb_rating and not pd.isna(tmdb_rating):
            combined_rating = (imdb_rating + tmdb_rating) / 2.0
        else:
            combined_rating = imdb_rating

        result = {
            'movie_id': int(movie_id),
            'title': movie['title'],
            'year': int(movie['year']),
            'runtime': int(movie['runtime']),
            'genres': list(movie['genres']) if isinstance(movie['genres'], (list, tuple)) else [],
            'avg_rating': float(imdb_rating),
            'tmdb_rating': float(tmdb_rating) if tmdb_rating and not pd.isna(tmdb_rating) else None,
            'combined_rating': float(combined_rating),
            'num_votes': int(movie['num_votes']),
            'director': movie['director'],
            'actors': list(movie['actors']) if isinstance(movie['actors'], (list, tuple)) else [],
            'country': movie.get('country'),
            'description': movie.get('description') if pd.notna(movie.get('description')) else None,
            'poster_url': movie.get('poster_url') if pd.notna(movie.get('poster_url')) else None,
            'backdrop_url': movie.get('backdrop_url') if pd.notna(movie.get('backdrop_url')) else None,
            'keywords': list(movie.get('keywords', [])) if movie.get('keywords') is not None else [],
            'budget': float(movie.get('budget', 0)) if pd.notna(movie.get('budget')) else None,
            'revenue': float(movie.get('revenue', 0)) if pd.notna(movie.get('revenue')) else None
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in get_movie: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    """
    Record user feedback and persist to database (remembers across sessions).

    Request:
    {
        "user_id": int,
        "session_id": str (optional, auto-generated if not provided),
        "movie_id": int,
        "action": "yes|no|final|left|right|up",
        "genres": [str] (optional, for context),
        "era": str (optional, for context),
        "themes": [str] (optional, for context),
        "position_in_session": int (optional),
        "previous_movie_id": int (optional)
    }

    Response:
    {
        "status": "ok"
    }
    """
    try:
        import time
        data = request.get_json()

        # Validate required fields
        required_fields = ['user_id', 'movie_id', 'action']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        user_id = data['user_id']
        movie_id = data['movie_id']
        action = data['action']
        session_id = data.get('session_id', f'session_{user_id}_{int(time.time())}')
        
        # Build context from request
        context = {
            'genres': data.get('genres', []),
            'era': data.get('era', 'any'),
            'themes': data.get('themes', [])
        }
        
        # Persist feedback to database
        engine.update_feedback(
            user_id=user_id,
            session_id=session_id,
            movie_id=movie_id,
            action=action,
            context=context,
            position_in_session=data.get('position_in_session', 0),
            previous_movie_id=data.get('previous_movie_id')
        )
        
        # Also log for debugging
        logger.info(f"Feedback persisted: user={user_id}, movie={movie_id}, action={action}, session={session_id}")
        
        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        logger.error(f"Error in record_feedback: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("SMART RECOMMENDATION API SERVER")
    logger.info("="*60)
    logger.info(f"Movies loaded: {len(engine.movies):,}")
    logger.info(f"Keyword database: {'✓ Loaded' if engine.keyword_analyzer else '✗ Not available'}")
    logger.info("")
    logger.info("Starting server on http://localhost:5000")
    logger.info("="*60)

    app.run(host='0.0.0.0', port=5000, debug=True)
