"""
Movie Pacing Calculation Module

Estimates movie pacing/intensity based on:
1. Genre composition
2. TMDb keywords
3. Runtime (future use)

Pacing score: 0.0 = very slow/contemplative, 1.0 = very fast/intense
"""

import pandas as pd
import numpy as np
from typing import List, Set


# Keywords that signal fast-paced movies
FAST_PACING_KEYWORDS: Set[str] = {
    'chase', 'heist', 'escape', 'survival', 'fast paced', 'fast-paced',
    'intense action', 'adrenaline', 'relentless', 'non-stop',
    'time pressure', 'race against time', 'pursuit', 'high-octane',
    'explosive', 'breakneck', 'frenetic'
}

# Keywords that signal slow-paced movies
SLOW_PACING_KEYWORDS: Set[str] = {
    'slow burn', 'slow-burn', 'contemplative', 'character study',
    'atmospheric', 'methodical', 'psychological', 'introspective',
    'meditative', 'deliberate', 'measured', 'pensive', 'reflective',
    'brooding', 'languid', 'cerebral'
}


def calculate_pacing_score(movie_row: pd.Series) -> float:
    """
    Calculate pacing score for a movie.

    Args:
        movie_row: Pandas Series with 'genres' and 'keywords' fields

    Returns:
        Float from 0.0 (slow) to 1.0 (fast)
    """
    score = 0.5  # Neutral baseline

    genres = movie_row.get('genres', [])
    keywords = movie_row.get('keywords', [])

    # Genre-based adjustments
    if genres is not None and len(genres) > 0:
        if isinstance(genres, (list, np.ndarray)):
            genre_lower = [str(g).lower() for g in genres]
        else:
            genre_lower = []

        if 'action' in genre_lower:
            score += 0.25
        if 'thriller' in genre_lower:
            score += 0.15
        if 'horror' in genre_lower:
            score += 0.10
        if 'drama' in genre_lower:
            score -= 0.20
        if 'documentary' in genre_lower:
            score -= 0.15
        if 'romance' in genre_lower:
            score -= 0.10

    # Keyword-based adjustments (stronger signals)
    if keywords is not None and len(keywords) > 0:
        if isinstance(keywords, (list, np.ndarray)):
            keyword_lower = [str(kw).lower() for kw in keywords]
        else:
            keyword_lower = []

        # Check for fast pacing keywords
        has_fast_keyword = any(
            fast_kw in kw_lower
            for kw_lower in keyword_lower
            for fast_kw in FAST_PACING_KEYWORDS
        )

        if has_fast_keyword:
            score += 0.25

        # Check for slow pacing keywords
        has_slow_keyword = any(
            slow_kw in kw_lower
            for kw_lower in keyword_lower
            for slow_kw in SLOW_PACING_KEYWORDS
        )

        if has_slow_keyword:
            score -= 0.25

    # Clamp to valid range
    return max(0.0, min(1.0, score))


def calculate_pacing_match(movie_row: pd.Series, pacing_preference: str) -> float:
    """
    Calculate how well a movie matches a pacing preference.

    Args:
        movie_row: Pandas Series with movie data
        pacing_preference: Either 'fast_paced' or 'slow_burn'

    Returns:
        Float from 0.0 to 1.0 indicating match quality
    """
    if not pacing_preference or pacing_preference not in ['fast_paced', 'slow_burn']:
        return 1.0  # No preference = no penalty

    movie_pacing = calculate_pacing_score(movie_row)

    if pacing_preference == 'fast_paced':
        # Want fast (0.7-1.0 ideal)
        if movie_pacing >= 0.7:
            return 1.0
        elif movie_pacing >= 0.5:
            return 0.6
        else:
            return 0.2

    elif pacing_preference == 'slow_burn':
        # Want slow (0.0-0.4 ideal)
        if movie_pacing <= 0.4:
            return 1.0
        elif movie_pacing <= 0.6:
            return 0.6
        else:
            return 0.2

    return 1.0


def precompute_pacing_scores(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'pacing_score' column to movies DataFrame.

    This should be run once during the data pipeline.

    Args:
        movies_df: DataFrame containing movie data

    Returns:
        DataFrame with added 'pacing_score' column
    """
    print("Computing pacing scores for all movies...")

    movies_df['pacing_score'] = movies_df.apply(calculate_pacing_score, axis=1)

    avg_pacing = movies_df['pacing_score'].mean()
    print(f"Average pacing score: {avg_pacing:.2f}")
    print(f"Fast-paced (>0.7): {(movies_df['pacing_score'] > 0.7).sum():,} movies")
    print(f"Slow-burn (<0.4): {(movies_df['pacing_score'] < 0.4).sum():,} movies")
    print(f"Medium (0.4-0.7): {((movies_df['pacing_score'] >= 0.4) & (movies_df['pacing_score'] <= 0.7)).sum():,} movies")

    return movies_df


def infer_pacing_from_session(session_history: List[dict]) -> str:
    """
    Infer pacing preference from user's session behavior.

    Args:
        session_history: List of movie dicts with 'action' and pacing info

    Returns:
        'fast', 'slow', or None
    """
    if not session_history or len(session_history) < 3:
        return None

    liked_movies = [m for m in session_history if m.get('action') == 'up']

    if len(liked_movies) < 3:
        return None

    # Calculate average pacing of liked movies
    pacing_scores = [m.get('pacing_score', 0.5) for m in liked_movies]
    avg_pacing = np.mean(pacing_scores)

    # Detect preference
    if avg_pacing > 0.7:
        return 'fast'
    elif avg_pacing < 0.4:
        return 'slow'
    else:
        return None


if __name__ == '__main__':
    # Standalone script to add pacing scores to existing movies.parquet
    import sys

    movies_path = 'output/processed/movies.parquet'

    print(f"Loading movies from {movies_path}")
    movies = pd.read_parquet(movies_path)

    print(f"Loaded {len(movies):,} movies")

    # Add pacing scores
    movies = precompute_pacing_scores(movies)

    # Save back to parquet
    print(f"\nSaving updated movies to {movies_path}")
    movies.to_parquet(movies_path, index=False)

    print("Done!")

    # Show some examples
    print("\n--- Fast-paced examples (score > 0.8) ---")
    fast_movies = movies[movies['pacing_score'] > 0.8].nlargest(5, 'avg_rating')
    for _, movie in fast_movies.iterrows():
        print(f"{movie['title']} ({movie['year']}) - Pacing: {movie['pacing_score']:.2f}, Rating: {movie['avg_rating']:.1f}")

    print("\n--- Slow-burn examples (score < 0.3) ---")
    slow_movies = movies[movies['pacing_score'] < 0.3].nlargest(5, 'avg_rating')
    for _, movie in slow_movies.iterrows():
        print(f"{movie['title']} ({movie['year']}) - Pacing: {movie['pacing_score']:.2f}, Rating: {movie['avg_rating']:.1f}")
