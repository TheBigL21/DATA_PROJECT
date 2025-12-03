"""
RECOMMENDATION API SERVER

Purpose: REST API for movie recommendation system.

Endpoints:
1. POST /api/recommend - Generate recommendations for user session
2. GET /api/movie/<movie_id> - Get detailed movie information
3. POST /api/feedback - Record user swipe feedback

This Flask server loads trained models and serves real-time recommendations.

Usage:
  python3 api.py

Then access at http://localhost:5000
"""

from flask import Flask, request, jsonify
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from recommendation.recommendation_engine import load_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load recommendation system
# Using demo models by default, change paths for production
MODELS_DIR = Path("./output_demo/models")
MOVIES_PATH = Path("./output_demo/movies.parquet")

logger.info("Loading recommendation system...")
engine = load_system(MODELS_DIR, MOVIES_PATH)
logger.info("System loaded successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        200 OK if system is operational
    """
    return jsonify({
        'status': 'ok',
        'num_movies': len(engine.movies),
        'num_users': engine.cf_model.num_users
    }), 200


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Generate movie recommendations for user session.

    Request body:
    {
        "user_id": int,
        "mood_genre": str,
        "mood_decade": str,
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
                "num_votes": int,
                "director": str,
                "actors": [str]
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['user_id', 'mood_genre', 'mood_decade']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        user_id = data['user_id']
        mood_genre = data['mood_genre']
        mood_decade = data['mood_decade']
        session_history = data.get('session_history', [])
        top_k = data.get('top_k', 20)

        logger.info(f"Recommendation request: user={user_id}, mood={mood_genre}/{mood_decade}, history_len={len(session_history)}")

        # Generate recommendations
        movie_ids = engine.recommend(
            user_id=user_id,
            mood_genre=mood_genre,
            mood_decade=mood_decade,
            session_history=session_history,
            top_k=top_k
        )

        # Get full movie details
        recommendations = []
        for movie_id in movie_ids:
            movie = engine.movie_dict[movie_id]
            rec = {
                'movie_id': int(movie['movie_id']),
                'title': str(movie['title']),
                'year': int(movie['year']),
                'genres': [str(g) for g in movie['genres']],
                'avg_rating': float(movie['avg_rating']),
                'num_votes': int(movie['num_votes']),
                'director': str(movie['director']),
                'actors': [str(a) for a in movie['actors']]
            }

            # Add TMDb data if available
            if 'description' in movie and movie['description'] is not None:
                rec['description'] = str(movie['description'])
            else:
                rec['description'] = None

            if 'poster_url' in movie and movie['poster_url'] is not None:
                rec['poster_url'] = str(movie['poster_url'])
            else:
                rec['poster_url'] = None

            recommendations.append(rec)

        logger.info(f"Returning {len(recommendations)} recommendations")

        return jsonify({
            'recommendations': recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/recommend: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """
    Get detailed information for a specific movie.

    Args:
        movie_id: Movie ID

    Response:
    {
        "movie_id": int,
        "title": str,
        "year": int,
        "runtime": int,
        "genres": [str],
        "avg_rating": float,
        "num_votes": int,
        "director": str,
        "actors": [str]
    }
    """
    try:
        if movie_id not in engine.movie_dict:
            return jsonify({'error': 'Movie not found'}), 404

        movie = engine.movie_dict[movie_id]

        result = {
            'movie_id': int(movie['movie_id']),
            'title': str(movie['title']),
            'year': int(movie['year']),
            'runtime': int(movie['runtime']),
            'genres': [str(g) for g in movie['genres']],
            'avg_rating': float(movie['avg_rating']),
            'num_votes': int(movie['num_votes']),
            'director': str(movie['director']),
            'actors': [str(a) for a in movie['actors']]
        }

        # Add TMDb data if available
        if 'description' in movie and movie['description'] is not None:
            result['description'] = str(movie['description'])
        else:
            result['description'] = None

        if 'poster_url' in movie and movie['poster_url'] is not None:
            result['poster_url'] = str(movie['poster_url'])
        else:
            result['poster_url'] = None

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in /api/movie/{movie_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Record user feedback (swipe action).

    Request body:
    {
        "user_id": int,
        "movie_id": int,
        "action": "left|right|up",
        "session_id": str,
        "timestamp": str (ISO format)
    }

    Response:
    {
        "status": "recorded"
    }

    Note: In production, this would write to database for model retraining.
    """
    try:
        data = request.get_json()

        required_fields = ['user_id', 'movie_id', 'action', 'session_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate action
        if data['action'] not in ['left', 'right', 'up']:
            return jsonify({'error': 'Invalid action, must be left|right|up'}), 400

        logger.info(f"Feedback recorded: user={data['user_id']}, movie={data['movie_id']}, action={data['action']}")

        # In production: save to database for model retraining
        # For now, just acknowledge

        return jsonify({
            'status': 'recorded'
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/feedback: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/moods', methods=['GET'])
def get_moods():
    """
    Get available mood options (genres and decades).

    Response:
    {
        "genres": [str],
        "decades": [str]
    }
    """
    try:
        # Extract unique genres from movies
        all_genres = sorted(set([g for genres in engine.movies['genres'] for g in genres]))

        # Standard decades
        decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

        return jsonify({
            'genres': all_genres,
            'decades': decades
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/moods: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_movies():
    """
    Search movies by title.

    Query parameters:
        q: search query (title substring)
        limit: max results (default 20)

    Response:
    {
        "results": [
            {
                "movie_id": int,
                "title": str,
                "year": int,
                "genres": [str],
                "avg_rating": float
            },
            ...
        ]
    }
    """
    try:
        query = request.args.get('q', '').lower()
        limit = int(request.args.get('limit', 20))

        if not query:
            return jsonify({'error': 'Missing query parameter: q'}), 400

        # Search by title substring
        results = []
        for _, movie in engine.movies.iterrows():
            if query in movie['title'].lower():
                results.append({
                    'movie_id': int(movie['movie_id']),
                    'title': str(movie['title']),
                    'year': int(movie['year']),
                    'genres': [str(g) for g in movie['genres']],
                    'avg_rating': float(movie['avg_rating'])
                })

                if len(results) >= limit:
                    break

        logger.info(f"Search query='{query}' returned {len(results)} results")

        return jsonify({
            'results': results
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/search: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MOVIE RECOMMENDATION API SERVER")
    print("="*60)
    print(f"Loaded {len(engine.movies)} movies")
    print(f"Loaded {engine.cf_model.num_users} users")
    print("\nAvailable endpoints:")
    print("  GET  /health")
    print("  POST /api/recommend")
    print("  GET  /api/movie/<movie_id>")
    print("  POST /api/feedback")
    print("  GET  /api/moods")
    print("  GET  /api/search?q=<query>")
    print("\nStarting server on http://localhost:8080")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=8080, debug=True)
