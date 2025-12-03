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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from recommendation.smart_engine import load_smart_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load recommendation system
MODELS_DIR = Path("./output/models")
MOVIES_PATH = Path("./output/processed/movies.parquet")
KEYWORD_DB_PATH = Path("./output/models/keyword_database.pkl")

logger.info("Loading smart recommendation system...")
try:
    engine = load_smart_system(MODELS_DIR, MOVIES_PATH, KEYWORD_DB_PATH)
    logger.info("✓ Smart system loaded successfully")
    logger.info(f"  - {len(engine.movies):,} movies")
    logger.info(f"  - Keyword analyzer: {'✓' if engine.keyword_analyzer else '✗'}")
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
                    'label': '🛋️ Chill Evening by myself',
                    'description': 'Relaxed solo viewing, open to experimenting'
                },
                {
                    'id': 'date_night',
                    'label': '💑 Date Night',
                    'description': 'Impressive movies for a special evening'
                },
                {
                    'id': 'family_night',
                    'label': '👨‍👩‍👧‍👦 Family Night',
                    'description': 'Safe, reliable entertainment for everyone'
                },
                {
                    'id': 'friends_night',
                    'label': '👥 Friends Night',
                    'description': 'Fun social viewing experience'
                }
            ],
            'genres': sorted(list(all_genres)),
            'eras': [
                {
                    'id': 'fresh',
                    'label': '🔥 Fresh Picks',
                    'description': 'Last 5 years (2020-2025)',
                    'years': '2020-2025'
                },
                {
                    'id': 'modern',
                    'label': '✨ Modern Classics',
                    'description': 'Last 10 years (2015-2025)',
                    'years': '2015-2025'
                },
                {
                    'id': 'timeless',
                    'label': '🎬 Timeless Favorites',
                    'description': 'Last 25 years (2000-2025)',
                    'years': '2000-2025'
                },
                {
                    'id': 'old_school',
                    'label': '🎞️ Old-School Gems',
                    'description': 'All time (1900-2025)',
                    'years': '1900-2025'
                }
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error in get_questionnaire_options: {e}")
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

        # Generate keywords
        if engine.keyword_analyzer:
            keywords = engine.suggest_keywords(genres, num_keywords=8)
        else:
            keywords = []

        return jsonify({'keywords': keywords}), 200

    except Exception as e:
        logger.error(f"Error in generate_keywords: {e}")
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
        "era": str,  // 'fresh', 'modern', 'timeless', 'old_school'
        "keywords": [str, str],  // 0-2 keywords (optional)
        "session_history": [
            {"movie_id": int, "action": "left|right|up"},
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
        session_history = data.get('session_history', [])
        top_k = data.get('top_k', 20)

        # Validate types
        if not isinstance(genres, list) or len(genres) == 0 or len(genres) > 2:
            return jsonify({'error': 'genres must be a list of 1-2 items'}), 400

        if keywords and not isinstance(keywords, list):
            return jsonify({'error': 'keywords must be a list'}), 400

        # Generate recommendations
        logger.info(f"Generating recommendations for user {user_id}")
        logger.info(f"  Evening: {evening_type}, Genres: {genres}, Era: {era}, Keywords: {keywords}")

        movie_ids = engine.recommend(
            user_id=user_id,
            evening_type=evening_type,
            genres=genres,
            era=era,
            keywords=keywords,
            session_history=session_history,
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
    Record user feedback (for future model updates).

    Request:
    {
        "user_id": int,
        "movie_id": int,
        "action": "left|right|up",
        "timestamp": int (optional)
    }

    Response:
    {
        "status": "ok"
    }
    """
    try:
        data = request.get_json()

        # Validate
        required_fields = ['user_id', 'movie_id', 'action']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Log feedback (in production, save to database)
        logger.info(f"Feedback: user={data['user_id']}, movie={data['movie_id']}, action={data['action']}")

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        logger.error(f"Error in record_feedback: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import pandas as pd

    # Add pandas import at top level
    pd.options.mode.chained_assignment = None

    logger.info("="*60)
    logger.info("SMART RECOMMENDATION API SERVER")
    logger.info("="*60)
    logger.info(f"Movies loaded: {len(engine.movies):,}")
    logger.info(f"Keyword database: {'✓ Loaded' if engine.keyword_analyzer else '✗ Not available'}")
    logger.info("")
    logger.info("Starting server on http://localhost:5000")
    logger.info("="*60)

    app.run(host='0.0.0.0', port=5000, debug=True)
