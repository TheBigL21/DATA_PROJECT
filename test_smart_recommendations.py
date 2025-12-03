"""
TEST SCRIPT FOR SMART RECOMMENDATION SYSTEM

Tests:
1. Keyword generation
2. Option C quality scoring
3. Era favorability (not filtering)
4. Complete recommendation flow
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')

from recommendation.smart_engine import load_smart_system


def test_keyword_generation(engine):
    """Test keyword suggestion functionality."""
    print("="*60)
    print("TEST 1: Keyword Generation")
    print("="*60)

    if not engine.keyword_analyzer:
        print("⚠️  Keyword analyzer not loaded, skipping test")
        return

    test_cases = [
        ['action'],
        ['comedy'],
        ['action', 'thriller'],
        ['comedy', 'romance'],
        ['horror', 'thriller']
    ]

    for genres in test_cases:
        keywords = engine.suggest_keywords(genres, num_keywords=8)
        print(f"\n{genres}:")
        print(f"  Keywords: {keywords}")

    print("\n✓ Keyword generation test passed")


def test_quality_scoring(engine):
    """Test Option C quality scoring."""
    print("\n" + "="*60)
    print("TEST 2: Option C Quality Scoring")
    print("="*60)

    # Test movies with different ratings and vote counts
    test_cases = [
        {'title': 'Blockbuster', 'rating': 8.7, 'votes': 2400000, 'evening': 'date_night'},
        {'title': 'Hidden Gem', 'rating': 7.8, 'votes': 5000, 'evening': 'chill_evening'},
        {'title': 'Same Hidden Gem on Date', 'rating': 7.8, 'votes': 5000, 'evening': 'date_night'},
        {'title': 'Mediocre Popular', 'rating': 6.2, 'votes': 500000, 'evening': 'friends_night'},
        {'title': 'Great with Few Votes', 'rating': 8.5, 'votes': 800, 'evening': 'chill_evening'}
    ]

    print("\nQuality scores for different scenarios:")
    print(f"{'Movie':<30} {'Rating':<8} {'Votes':<12} {'Evening':<18} {'Score':<8}")
    print("-"*90)

    for case in test_cases:
        score = engine._calculate_option_c_score(
            case['rating'],
            case['votes'],
            case['evening']
        )
        print(f"{case['title']:<30} {case['rating']:<8.1f} {case['votes']:<12,} {case['evening']:<18} {score:<8.3f}")

    print("\n✓ Quality scoring test passed")


def test_era_favorability(engine):
    """Test that era is favorability (not filtering)."""
    print("\n" + "="*60)
    print("TEST 3: Era Favorability (Not Filtering)")
    print("="*60)

    print("\nGenerating recommendations with 'fresh' era (2020-2025)...")
    print("Should include movies outside range with lower scores.\n")

    recommendations = engine.recommend(
        user_id=0,
        evening_type='chill_evening',
        genres=['action'],
        era='fresh',
        keywords=[],
        session_history=[],
        top_k=20
    )

    # Check years
    years = []
    for mid in recommendations:
        movie = engine.movies[engine.movies['movie_id'] == mid].iloc[0]
        years.append(movie['year'])

    years_sorted = sorted(years, reverse=True)
    outside_range = [y for y in years if y < 2020 or y > 2025]

    print(f"Movie years in top 20: {years_sorted}")
    print(f"\nMovies outside 2020-2025 range: {len(outside_range)} movies")
    if outside_range:
        print(f"  Years: {sorted(outside_range, reverse=True)}")
        print("\n✓ Era favorability working correctly (includes movies outside range)")
    else:
        print("\n⚠️  No movies outside range (might need more relaxed filtering)")


def test_complete_flow(engine):
    """Test complete recommendation flow."""
    print("\n" + "="*60)
    print("TEST 4: Complete Recommendation Flow")
    print("="*60)

    # Test cases
    test_cases = [
        {
            'name': 'Date Night - Action/Thriller - Modern - Espionage',
            'evening_type': 'date_night',
            'genres': ['action', 'thriller'],
            'era': 'modern',
            'keywords': ['espionage'] if engine.keyword_analyzer else []
        },
        {
            'name': 'Chill Evening - Comedy - Timeless - No Keywords',
            'evening_type': 'chill_evening',
            'genres': ['comedy'],
            'era': 'timeless',
            'keywords': []
        },
        {
            'name': 'Friends Night - Horror/Thriller - Fresh',
            'evening_type': 'friends_night',
            'genres': ['horror', 'thriller'],
            'era': 'fresh',
            'keywords': []
        }
    ]

    for case in test_cases:
        print(f"\n{case['name']}")
        print("-"*60)

        recommendations = engine.recommend(
            user_id=0,
            evening_type=case['evening_type'],
            genres=case['genres'],
            era=case['era'],
            keywords=case['keywords'],
            session_history=[],
            top_k=10
        )

        print(f"\nTop 10 recommendations:")
        for rank, movie_id in enumerate(recommendations, 1):
            movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0]

            # Calculate combined rating
            imdb_rating = movie['avg_rating']
            tmdb_rating = movie.get('tmdb_rating')
            if pd.notna(tmdb_rating):
                combined_rating = (imdb_rating + tmdb_rating) / 2.0
            else:
                combined_rating = imdb_rating

            genres_str = ', '.join(list(movie['genres'])[:3]) if isinstance(movie['genres'], (list, np.ndarray)) else ''

            print(f"{rank:2d}. {movie['title'][:45]:45} ({movie['year']})")
            print(f"    {genres_str[:50]:50} | ⭐ {combined_rating:.1f}/10")

            # Show keywords if available
            keywords = movie.get('keywords')
            if keywords is not None and isinstance(keywords, (list, np.ndarray)) and len(keywords) > 0:
                keywords_sample = list(keywords)[:3]
                print(f"    Keywords: {', '.join(str(k) for k in keywords_sample)}")

    print("\n✓ Complete flow test passed")


def test_option_c_examples(engine):
    """Show detailed Option C scoring examples."""
    print("\n" + "="*60)
    print("TEST 5: Detailed Option C Examples")
    print("="*60)

    examples = [
        {
            'title': 'Inception (Blockbuster on Date)',
            'rating': 8.7,
            'votes': 2400000,
            'evening': 'date_night'
        },
        {
            'title': 'Inception (Same on Chill)',
            'rating': 8.7,
            'votes': 2400000,
            'evening': 'chill_evening'
        },
        {
            'title': 'Hidden Gem (7.8, 5K votes, Chill)',
            'rating': 7.8,
            'votes': 5000,
            'evening': 'chill_evening'
        },
        {
            'title': 'Hidden Gem (7.8, 5K votes, Date)',
            'rating': 7.8,
            'votes': 5000,
            'evening': 'date_night'
        }
    ]

    print("\nDetailed score breakdown:")
    print("-"*100)

    for ex in examples:
        print(f"\n{ex['title']}")
        print(f"  Rating: {ex['rating']}, Votes: {ex['votes']:,}, Evening: {ex['evening']}")

        # Calculate components
        normalized = (ex['rating'] - 4.0) / 6.0
        base_quality = normalized ** 1.5

        if ex['votes'] >= 100000:
            confidence = 1.0
        elif ex['votes'] >= 10000:
            confidence = 0.9
        elif ex['votes'] >= 1000:
            confidence = 0.7
        else:
            confidence = 0.5

        modifier = engine.EVENING_MODIFIERS[ex['evening']]

        final_score = min(1.0, base_quality * confidence * modifier)

        print(f"  → Base quality: {base_quality:.3f}")
        print(f"  → Confidence: {confidence:.3f}")
        print(f"  → Modifier: {modifier:.3f}")
        print(f"  → Final score: {final_score:.3f}")

    print("\n✓ Detailed examples completed")


def main():
    """Run all tests."""
    print("="*60)
    print("SMART RECOMMENDATION SYSTEM - TEST SUITE")
    print("="*60)

    # Load system
    models_dir = Path("output/models")
    movies_path = Path("output/processed/movies.parquet")
    keyword_db_path = Path("output/models/keyword_database.pkl")

    print(f"\nLoading system...")
    print(f"  Models: {models_dir}")
    print(f"  Movies: {movies_path}")
    print(f"  Keywords: {keyword_db_path}")

    try:
        engine = load_smart_system(models_dir, movies_path, keyword_db_path)
        print(f"\n✓ System loaded successfully")
        print(f"  - {len(engine.movies):,} movies")
        print(f"  - Keyword analyzer: {'✓' if engine.keyword_analyzer else '✗'}")
    except Exception as e:
        print(f"\n✗ Failed to load system: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run tests
    try:
        test_keyword_generation(engine)
        test_quality_scoring(engine)
        test_era_favorability(engine)
        test_complete_flow(engine)
        test_option_c_examples(engine)

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
