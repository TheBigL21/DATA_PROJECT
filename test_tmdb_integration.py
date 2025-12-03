"""
Quick test script to verify TMDb integration works correctly.
Tests the complete flow: fetch TMDb data → merge with IMDb → verify output
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.transform_imdb_data import load_imdb_tables, build_internal_movies_table


def test_tmdb_integration():
    """Test that TMDb data is properly merged into final movies table."""

    print("="*60)
    print("TESTING TMDb INTEGRATION")
    print("="*60)

    # Check if TMDb data exists
    tmdb_path = Path('output/tmdb_enrichment.parquet')
    if not tmdb_path.exists():
        print("\n❌ ERROR: tmdb_enrichment.parquet not found!")
        print("Run: python3 src/data_processing/fetch_tmdb_data.py output/movies_core.parquet output <api_key> 20")
        return False

    # Load TMDb data
    tmdb_df = pd.read_parquet(tmdb_path)
    print(f"\n✓ Found TMDb enrichment: {len(tmdb_df)} movies")

    # Load IMDb tables
    print("\n1. Loading IMDb tables...")
    tables = load_imdb_tables(Path('output'))

    # Build internal movies table (should merge TMDb data)
    print("\n2. Building internal movies table with TMDb merge...")
    movies = build_internal_movies_table(tables)

    # Verify TMDb columns exist
    print("\n3. Verifying TMDb columns...")
    tmdb_columns = [
        'tmdb_id', 'tmdb_popularity', 'tmdb_rating', 'tmdb_vote_count',
        'description', 'poster_url', 'backdrop_url', 'keywords', 'budget', 'revenue'
    ]

    missing_cols = [col for col in tmdb_columns if col not in movies.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False

    print("✓ All TMDb columns present")

    # Check how many movies have TMDb data
    movies_with_tmdb = movies['tmdb_id'].notna().sum()
    print(f"\n✓ {movies_with_tmdb} / {len(movies)} movies have TMDb data")

    # Show example movie with TMDb data
    example = movies[movies['tmdb_id'].notna()].iloc[0]
    print("\n4. Example movie with TMDb data:")
    print(f"   Title: {example['title']} ({example['year']})")
    print(f"   IMDb: {example['tconst']} (rating: {example['avg_rating']}, votes: {example['num_votes']})")
    print(f"   TMDb: {int(example['tmdb_id'])} (rating: {example['tmdb_rating']}, votes: {example['tmdb_vote_count']})")
    print(f"   Description: {example['description'][:120]}...")
    print(f"   Poster: {example['poster_url']}")
    print(f"   Keywords: {example['keywords']}")
    print(f"   Budget: ${example['budget']:,}")
    print(f"   Revenue: ${example['revenue']:,}")

    # Compare IMDb vs TMDb ratings
    movies_with_both = movies[movies['tmdb_rating'].notna()]
    if len(movies_with_both) > 0:
        print(f"\n5. Rating comparison (for {len(movies_with_both)} movies):")
        print(f"   Avg IMDb rating: {movies_with_both['avg_rating'].mean():.2f}")
        print(f"   Avg TMDb rating: {movies_with_both['tmdb_rating'].mean():.2f}")
        print(f"   Correlation: {movies_with_both['avg_rating'].corr(movies_with_both['tmdb_rating']):.3f}")

    print("\n" + "="*60)
    print("✓ TMDb INTEGRATION TEST PASSED")
    print("="*60)
    print("\nNext steps:")
    print("1. Fetch more TMDb data (remove max_movies limit)")
    print("2. Use combined ratings in recommendation model")
    print("3. Use descriptions for content-based filtering")
    print("4. Use poster URLs in frontend application")

    return True


if __name__ == '__main__':
    success = test_tmdb_integration()
    sys.exit(0 if success else 1)
