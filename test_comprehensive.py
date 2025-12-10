"""
COMPREHENSIVE TEST SCRIPT FOR MOVIE FINDER
Tests multiple different input scenarios to ensure the system works correctly
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from recommendation.smart_engine import load_smart_system

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_recommendations(engine, test_name, **kwargs):
    """
    Test the recommendation engine with given parameters

    Args:
        engine: SmartRecommendationEngine instance
        test_name: Name of the test case
        **kwargs: Parameters to pass to recommend()
    """
    print(f"Test: {test_name}")
    print("-" * 80)
    print(f"Parameters:")
    for key, value in kwargs.items():
        if key not in ['user_id', 'top_k']:
            print(f"  {key}: {value}")
    print()

    try:
        recommendations = engine.recommend(
            user_id=0,
            **kwargs,
            top_k=5  # Get top 5 for testing
        )

        if recommendations and len(recommendations) > 0:
            print(f"✓ SUCCESS - Found {len(recommendations)} recommendations:\n")
            for i, movie_id in enumerate(recommendations, 1):
                # Convert movie_id to movie dict
                movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0].to_dict()

                title = movie.get('title', movie.get('primaryTitle', 'Unknown'))
                year = movie.get('year', movie.get('startYear', 'N/A'))
                rating = movie.get('avg_rating', movie.get('averageRating', 0))
                genres = movie.get('genres', [])
                if isinstance(genres, (list, tuple)):
                    genres_str = ', '.join([g.capitalize() for g in genres[:3]])
                else:
                    genres_str = str(genres)

                print(f"  {i}. {title} ({year}) - Rating: {rating:.1f}/10")
                print(f"     Genres: {genres_str}")
                print()
            return True
        else:
            print("⚠ WARNING - No recommendations found")
            return False

    except Exception as e:
        print(f"✗ ERROR - {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_section("COMPREHENSIVE MOVIE FINDER TEST SUITE")

    # Initialize the engine using the proper loader
    print("Loading recommendation system...")
    try:
        models_dir = Path('./output/models')
        movies_path = Path('./output/processed/movies.parquet')
        keyword_db_path = Path('./output/models/keyword_database.pkl')

        engine = load_smart_system(models_dir, movies_path, keyword_db_path)

        # Get movie count
        movie_count = len(engine.movies)
        print(f"✓ System loaded with {movie_count} movies\n")
    except Exception as e:
        print(f"✗ ERROR loading system: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cases matching the actual recommend() function signature
    # NOTE: Genres must be lowercase to match database
    test_cases = [
        {
            'name': 'Action Movies - Fresh Era',
            'params': {
                'genres': ['action'],
                'era': 'fresh',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Drama & Romance - Modern Era',
            'params': {
                'genres': ['drama', 'romance'],
                'era': 'modern',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Comedy - Timeless',
            'params': {
                'genres': ['comedy'],
                'era': 'timeless',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Thriller - Old School',
            'params': {
                'genres': ['thriller'],
                'era': 'old_school',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Sci-Fi - Based on Books',
            'params': {
                'genres': ['sci-fi'],
                'era': 'modern',
                'source_material': 'book',
                'themes': None
            }
        },
        {
            'name': 'Horror - Any Era',
            'params': {
                'genres': ['horror'],
                'era': None,
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Biography - Based on True Story',
            'params': {
                'genres': ['biography'],
                'era': None,
                'source_material': 'true_story',
                'themes': None
            }
        },
        {
            'name': 'Animation & Family - Fresh',
            'params': {
                'genres': ['animation', 'family'],
                'era': 'fresh',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Mystery & Crime - Modern',
            'params': {
                'genres': ['mystery', 'crime'],
                'era': 'modern',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'War Movies - Old School',
            'params': {
                'genres': ['war'],
                'era': 'old_school',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Fantasy - Original Screenplay',
            'params': {
                'genres': ['fantasy'],
                'era': None,
                'source_material': 'original',
                'themes': None
            }
        },
        {
            'name': 'Documentary - Fresh',
            'params': {
                'genres': ['documentary'],
                'era': 'fresh',
                'source_material': 'any',
                'themes': None
            }
        },
        {
            'name': 'Action - Based on Comics',
            'params': {
                'genres': ['action'],
                'era': 'modern',
                'source_material': 'comic',
                'themes': None
            }
        },
        {
            'name': 'Drama - Based on Play/Musical',
            'params': {
                'genres': ['drama'],
                'era': None,
                'source_material': 'play_musical',
                'themes': None
            }
        }
    ]

    # Run all tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        success = test_recommendations(engine, test_case['name'], **test_case['params'])
        results.append({'test': test_case['name'], 'success': success})

    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed

    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%\n")

    if failed > 0:
        print("Failed Tests:")
        for r in results:
            if not r['success']:
                print(f"  - {r['test']}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
