"""
Book/Novel Adaptation Detection Module

Detects whether a movie is adapted from a book or novel using:
1. TMDb keywords field
2. Movie description text analysis
"""

import pandas as pd
import numpy as np
from typing import List, Set


# Keywords that indicate book/novel adaptation
BOOK_KEYWORDS: Set[str] = {
    'based on novel',
    'based on book',
    'literary adaptation',
    'novel',
    'book adaptation',
    'adapted from novel',
    'based on the novel',
    'based on the book',
    'book',
}


def is_book_adaptation(movie_row: pd.Series) -> bool:
    """
    Check if a movie is adapted from a book or novel.

    Args:
        movie_row: Pandas Series containing movie data with 'keywords' and 'description' fields

    Returns:
        True if movie is a book adaptation, False otherwise
    """
    # Check TMDb keywords field
    keywords = movie_row.get('keywords', [])
    if keywords is not None and len(keywords) > 0:
        if isinstance(keywords, (list, np.ndarray)):
            keywords_lower = [str(kw).lower() for kw in keywords]
            if any(book_kw in keywords_lower for book_kw in BOOK_KEYWORDS):
                return True

    # Check description as fallback
    description = movie_row.get('description', '')
    if description and isinstance(description, str):
        desc_lower = description.lower()

        # Look for explicit phrases
        explicit_phrases = [
            'based on the novel',
            'based on the book',
            'adapted from the novel',
            'adapted from the book',
            'from the novel',
            'from the book'
        ]

        if any(phrase in desc_lower for phrase in explicit_phrases):
            return True

    return False


def precompute_adaptation_flags(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'is_book_adaptation' boolean column to movies DataFrame.

    This should be run once during the data pipeline to pre-compute
    adaptation flags for all movies.

    Args:
        movies_df: DataFrame containing movie data

    Returns:
        DataFrame with added 'is_book_adaptation' column
    """
    print("Computing book adaptation flags for all movies...")

    movies_df['is_book_adaptation'] = movies_df.apply(is_book_adaptation, axis=1)

    num_adaptations = movies_df['is_book_adaptation'].sum()
    total_movies = len(movies_df)
    percentage = (num_adaptations / total_movies) * 100

    print(f"Found {num_adaptations:,} adaptations out of {total_movies:,} movies ({percentage:.1f}%)")

    return movies_df


if __name__ == '__main__':
    # Standalone script to add adaptation flags to existing movies.parquet
    import sys

    movies_path = 'output/processed/movies.parquet'

    print(f"Loading movies from {movies_path}")
    movies = pd.read_parquet(movies_path)

    print(f"Loaded {len(movies):,} movies")

    # Add adaptation flags
    movies = precompute_adaptation_flags(movies)

    # Save back to parquet
    print(f"Saving updated movies to {movies_path}")
    movies.to_parquet(movies_path, index=False)

    print("Done!")
