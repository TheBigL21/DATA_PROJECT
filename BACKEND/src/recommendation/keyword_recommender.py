"""
KEYWORD RECOMMENDER - Genre-based keyword suggestions

Recommends relevant thematic keywords based on selected genre(s) to help users
refine their movie search. Now with comprehensive filtering.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple
import sys

# Import the keyword filter
sys.path.insert(0, str(Path(__file__).parent.parent))
from recommendation.keyword_filter import KeywordFilter


class KeywordRecommender:
    """Recommends keywords based on genre selection"""

    def __init__(self, movies_path: str = 'output/processed/movies.parquet'):
        """Initialize with movie data"""
        self.df = pd.read_parquet(movies_path)
        self.genre_keywords = self._build_genre_keyword_map()

    def _build_genre_keyword_map(self) -> Dict[str, Counter]:
        """Build map of genres to keyword frequencies"""
        import numpy as np
        genre_keywords = {}

        for idx, row in self.df.iterrows():
            # Skip if missing keywords or genres
            if not self._has_valid_data(row['keywords']) or not self._has_valid_data(row['genres']):
                continue

            # Handle numpy arrays and lists
            keywords = list(row['keywords']) if isinstance(row['keywords'], (list, tuple, np.ndarray)) else []
            genres = list(row['genres']) if isinstance(row['genres'], (list, tuple, np.ndarray)) else []

            for genre in genres:
                if genre not in genre_keywords:
                    genre_keywords[genre] = Counter()
                genre_keywords[genre].update(keywords)

        return genre_keywords

    def _has_valid_data(self, value) -> bool:
        """Check if value is valid (not None, not NaN)"""
        if value is None:
            return False
        if isinstance(value, float):
            import numpy as np
            return not np.isnan(value)
        return True

    def get_keywords_for_genres(
        self,
        genres: List[str],
        num_keywords: int = 8,
        exclude_generic: bool = True
    ) -> List[str]:
        """
        Get recommended keywords for given genre(s)

        Args:
            genres: List of 1-2 genres
            num_keywords: Number of keywords to return (default 8)
            exclude_generic: Whether to exclude overly generic keywords

        Returns:
            List of recommended keywords (thematically relevant only)
        """
        if len(genres) == 1:
            # Single genre - get top keywords
            genre = genres[0].lower()
            if genre not in self.genre_keywords:
                return []

            keywords = self.genre_keywords[genre].most_common(num_keywords * 5)

        else:
            # Multiple genres - find keywords common to both/intersection + unique to each
            keywords = self._get_multi_genre_keywords(genres, num_keywords * 5)

        # Filter using comprehensive KeywordFilter
        selected = []
        for keyword, count in keywords:
            # Use the new comprehensive filter
            if KeywordFilter.is_relevant(keyword):
                selected.append(keyword)
                if len(selected) >= num_keywords:
                    break

        return selected[:num_keywords]

    def _get_multi_genre_keywords(
        self,
        genres: List[str],
        total_needed: int
    ) -> List[Tuple[str, int]]:
        """
        Get keywords for multiple genres by combining:
        1. Keywords common to both genres (intersection)
        2. Top keywords from each genre
        """
        genre_counters = []
        for genre in genres:
            g = genre.lower()
            if g in self.genre_keywords:
                genre_counters.append(self.genre_keywords[g])

        if not genre_counters:
            return []

        # Find intersection - keywords that appear in all selected genres
        common_keywords = set(genre_counters[0].keys())
        for counter in genre_counters[1:]:
            common_keywords &= set(counter.keys())

        # Score common keywords by sum of frequencies
        combined = Counter()
        for keyword in common_keywords:
            combined[keyword] = sum(counter[keyword] for counter in genre_counters)

        # Add unique keywords from each genre
        for counter in genre_counters:
            for keyword, count in counter.most_common(20):
                if keyword not in combined:
                    combined[keyword] = count

        return combined.most_common(total_needed)

    def get_keyword_examples(self, genre1: str, genre2: str = None) -> Dict:
        """
        Get example keywords with descriptions for documentation

        Returns dict with genre(s) and example keywords with descriptions
        """
        genres = [genre1] if not genre2 else [genre1, genre2]
        keywords = self.get_keywords_for_genres(genres, num_keywords=6)

        # Get sample movies for each keyword
        examples = {
            'genres': genres,
            'keywords': []
        }

        for keyword in keywords:
            # Find movies with this keyword and genre(s)
            sample_movies = self._get_sample_movies_for_keyword(keyword, genres)
            examples['keywords'].append({
                'keyword': keyword,
                'sample_movies': sample_movies[:3]  # Top 3 examples
            })

        return examples

    def _get_sample_movies_for_keyword(
        self,
        keyword: str,
        genres: List[str],
        limit: int = 3
    ) -> List[str]:
        """Get sample movie titles that match keyword and genres"""
        import numpy as np
        matches = []

        for idx, row in self.df.iterrows():
            if not self._has_valid_data(row['keywords']) or not self._has_valid_data(row['genres']):
                continue

            keywords = list(row['keywords']) if isinstance(row['keywords'], (list, tuple, np.ndarray)) else []
            movie_genres = list(row['genres']) if isinstance(row['genres'], (list, tuple, np.ndarray)) else []

            # Check if keyword matches and at least one genre matches
            if keyword in keywords and any(g in movie_genres for g in genres):
                matches.append(f"{row['title']} ({row['year']})")
                if len(matches) >= limit:
                    break

        return matches


def generate_all_examples():
    """Generate example keywords for all major genre combinations"""
    recommender = KeywordRecommender()

    # Major genres to demonstrate
    major_genres = [
        'action', 'comedy', 'drama', 'horror', 'thriller',
        'romance', 'sci-fi', 'fantasy', 'crime', 'adventure'
    ]

    print("=" * 70)
    print("KEYWORD RECOMMENDATION EXAMPLES")
    print("=" * 70)

    # Single genres
    print("\n📌 SINGLE GENRE EXAMPLES\n")
    for genre in major_genres[:6]:
        keywords = recommender.get_keywords_for_genres([genre], num_keywords=6)
        print(f"\n{genre.upper()}:")
        for i, kw in enumerate(keywords, 1):
            print(f"  {i}. {kw}")

    # Genre combinations
    print("\n\n📌 GENRE COMBINATION EXAMPLES\n")
    combinations = [
        ('action', 'thriller'),
        ('comedy', 'romance'),
        ('sci-fi', 'horror'),
        ('drama', 'crime'),
        ('fantasy', 'adventure'),
        ('horror', 'mystery')
    ]

    for genre1, genre2 in combinations:
        keywords = recommender.get_keywords_for_genres([genre1, genre2], num_keywords=6)
        print(f"\n{genre1.upper()} + {genre2.upper()}:")
        for i, kw in enumerate(keywords, 1):
            print(f"  {i}. {kw}")

    # Save to JSON
    all_examples = {
        'single_genres': {},
        'combinations': {}
    }

    for genre in major_genres:
        keywords = recommender.get_keywords_for_genres([genre], num_keywords=6)
        all_examples['single_genres'][genre] = keywords

    for genre1, genre2 in combinations:
        keywords = recommender.get_keywords_for_genres([genre1, genre2], num_keywords=6)
        all_examples['combinations'][f"{genre1}+{genre2}"] = keywords

    output_path = 'config/keyword_examples.json'
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)

    print(f"\n\n✅ Saved examples to {output_path}")


if __name__ == '__main__':
    generate_all_examples()
