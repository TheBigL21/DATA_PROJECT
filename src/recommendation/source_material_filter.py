"""
SOURCE MATERIAL FILTER MODULE

Handles filtering movies by their source material (book, true story, original, etc.)
Detects source material from TMDB keywords.
"""

from typing import List, Optional, Set
import numpy as np


class SourceMaterialFilter:
    """Manages source material filtering for movie recommendations"""

    # Source material keyword mappings
    SOURCE_KEYWORDS = {
        'book': {
            'keywords': {
                'based on novel', 'based on book', 'based on novel or book',
                'literary adaptation', 'novel', 'book adaptation'
            },
            'label': 'Based on Book/Novel',
            'short_label': 'Book',
            'description': 'Movies adapted from books or novels'
        },
        'true_story': {
            'keywords': {
                'based on true story', 'based on real events', 'true story',
                'biography', 'biographical', 'biopic', 'real life',
                'historical figure'
            },
            'label': 'Based on True Story',
            'short_label': 'True Story',
            'description': 'Movies based on real events or people'
        },
        'play_musical': {
            'keywords': {
                'based on play', 'based on play or musical', 'based on musical',
                'stage adaptation', 'theatrical adaptation', 'broadway'
            },
            'label': 'Based on Play/Musical',
            'short_label': 'Play/Musical',
            'description': 'Movies adapted from theatrical works'
        },
        'comic': {
            'keywords': {
                'based on comic', 'based on comic book', 'comic book',
                'based on manga', 'manga', 'graphic novel', 'superhero'
            },
            'label': 'Based on Comic/Manga',
            'short_label': 'Comic',
            'description': 'Movies adapted from comics or manga'
        },
        'original': {
            'keywords': set(),  # Special case - no keywords means original
            'label': 'Original Screenplay',
            'short_label': 'Original',
            'description': 'Original stories not based on existing material'
        },
        'any': {
            'keywords': set(),
            'label': "Doesn't Matter",
            'short_label': 'Any',
            'description': 'No preference on source material'
        }
    }

    @classmethod
    def get_source_options(cls) -> list:
        """
        Get list of source material options for display.

        Returns:
            List of dicts with source info
        """
        return [
            {
                'id': source_id,
                'label': info['label'],
                'short_label': info['short_label'],
                'description': info['description']
            }
            for source_id, info in cls.SOURCE_KEYWORDS.items()
        ]

    @classmethod
    def detect_source_material(cls, movie_keywords: List[str]) -> Set[str]:
        """
        Detect source material types from movie keywords.

        Args:
            movie_keywords: List of keywords for a movie

        Returns:
            Set of detected source types (e.g., {'book', 'true_story'})
        """
        # Handle empty/null values properly (including numpy arrays)
        if movie_keywords is None or len(movie_keywords) == 0:
            return {'original'}

        # Normalize keywords
        movie_kw_lower = {str(kw).lower().strip() for kw in movie_keywords}
        detected = set()

        for source_id, source_info in cls.SOURCE_KEYWORDS.items():
            if source_id in ['original', 'any']:
                continue

            # Check if any source keywords match
            if movie_kw_lower & source_info['keywords']:
                detected.add(source_id)

        # If nothing detected, it's original
        if not detected:
            detected.add('original')

        return detected

    @classmethod
    def matches_source_preference(cls, movie_keywords: List[str],
                                   preferred_source: str) -> bool:
        """
        Check if a movie matches the preferred source material.

        Args:
            movie_keywords: List of keywords for a movie
            preferred_source: Preferred source ID (e.g., 'book', 'true_story', 'any')

        Returns:
            True if movie matches preference
        """
        if preferred_source == 'any':
            return True

        detected_sources = cls.detect_source_material(movie_keywords)

        return preferred_source in detected_sources

    @classmethod
    def calculate_source_score(cls, movie_keywords: List[str],
                                preferred_source: str) -> float:
        """
        Calculate how well a movie matches source preference.

        Args:
            movie_keywords: List of keywords for a movie
            preferred_source: Preferred source ID

        Returns:
            Score between 0.0 and 1.0
        """
        if preferred_source == 'any':
            return 1.0  # No preference

        detected_sources = cls.detect_source_material(movie_keywords)

        if preferred_source in detected_sources:
            return 1.0
        else:
            # Partial penalty for not matching
            return 0.3

    @classmethod
    def filter_movies_by_source(cls, movies: list, preferred_source: str,
                                 strict: bool = False) -> list:
        """
        Filter movies by source material.

        Args:
            movies: List of movie dicts (must have 'keywords' field)
            preferred_source: Preferred source ID
            strict: If True, only return matching movies. If False, score all.

        Returns:
            Filtered/scored list of movies
        """
        if preferred_source == 'any':
            return movies

        filtered = []

        for movie in movies:
            keywords = movie.get('keywords', [])

            # Handle numpy arrays
            if isinstance(keywords, np.ndarray):
                keywords = keywords.tolist()

            matches = cls.matches_source_preference(keywords, preferred_source)

            if strict:
                if matches:
                    filtered.append(movie)
            else:
                source_score = cls.calculate_source_score(keywords, preferred_source)
                movie['source_score'] = source_score
                filtered.append(movie)

        return filtered

    @classmethod
    def get_source_label(cls, source_id: str) -> str:
        """Get display label for a source type"""
        if source_id in cls.SOURCE_KEYWORDS:
            return cls.SOURCE_KEYWORDS[source_id]['label']
        return "Unknown Source"

    @classmethod
    def get_source_short_label(cls, source_id: str) -> str:
        """Get short display label for a source type"""
        if source_id in cls.SOURCE_KEYWORDS:
            return cls.SOURCE_KEYWORDS[source_id]['short_label']
        return "Unknown"


if __name__ == '__main__':
    """Test the source material filter"""
    print("="*70)
    print("SOURCE MATERIAL FILTER TEST")
    print("="*70)

    # Display all source options
    print("\nAvailable Source Material Options:\n")
    for i, source in enumerate(SourceMaterialFilter.get_source_options(), 1):
        print(f"  {i}. {source['label']}")
        print(f"     {source['description']}\n")

    # Test source detection
    print(f"{'='*70}")
    print("SOURCE DETECTION TEST")
    print(f"{'='*70}\n")

    test_cases = [
        {
            'title': 'The Lord of the Rings',
            'keywords': ['based on novel', 'fantasy', 'epic'],
            'expected': {'book'}
        },
        {
            'title': 'The Social Network',
            'keywords': ['based on true story', 'biography', 'facebook'],
            'expected': {'true_story'}
        },
        {
            'title': 'The Avengers',
            'keywords': ['based on comic', 'superhero', 'marvel'],
            'expected': {'comic'}
        },
        {
            'title': 'Inception',
            'keywords': ['dream', 'science fiction', 'heist'],
            'expected': {'original'}
        },
        {
            'title': 'West Side Story',
            'keywords': ['based on play or musical', 'musical', 'romance'],
            'expected': {'play_musical'}
        },
    ]

    print("Testing source material detection:\n")
    for test in test_cases:
        detected = SourceMaterialFilter.detect_source_material(test['keywords'])
        match = detected == test['expected']
        status = "✓" if match else "✗"
        print(f"  {status} {test['title']}")
        print(f"     Detected: {detected}")
        print(f"     Expected: {test['expected']}\n")

    # Test source scoring
    print(f"{'='*70}")
    print("SOURCE SCORING TEST")
    print(f"{'='*70}\n")

    test_movies = [
        {'title': 'Harry Potter', 'keywords': ['based on novel', 'magic', 'wizard']},
        {'title': 'Inception', 'keywords': ['dream', 'heist']},
        {'title': 'Spider-Man', 'keywords': ['based on comic', 'superhero']},
    ]

    preferred = 'book'
    print(f"Movies scored for preference '{preferred}':\n")

    for movie in test_movies:
        score = SourceMaterialFilter.calculate_source_score(
            movie['keywords'], preferred
        )
        detected = SourceMaterialFilter.detect_source_material(movie['keywords'])
        print(f"  {movie['title']}: {score:.1f}")
        print(f"    Sources: {detected}\n")
