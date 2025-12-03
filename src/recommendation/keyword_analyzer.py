"""
KEYWORD ANALYZER MODULE

Analyzes movie descriptions and TMDB keywords to build contextual
keyword recommendations for genre combinations.

Features:
- Extract top keywords per genre
- Extract unique keywords per genre combination
- Filter generic/uninformative keywords
- Calculate TF-IDF relevance scores
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set
import re
import pickle
from pathlib import Path


class KeywordAnalyzer:
    """
    Analyzes movies to extract contextually relevant keywords
    for genre-based recommendations.
    """

    # Generic keywords to exclude (low information value)
    GENERIC_KEYWORDS = {
        # Technical/Format
        'based on novel', 'based on book', 'based on play', 'based on comic',
        'remake', 'sequel', 'prequel', 'reboot', 'spin off',
        'black and white', 'color', 'silent film', 'independent film',

        # Generic descriptors
        'low budget', 'cult film', 'cult classic', 'surprise ending',
        'flashback', 'voice over narration', 'narration',

        # Too broad/vague
        'male protagonist', 'female protagonist', 'protagonist',
        'good versus evil', 'violence', 'death', 'murder',
        'friendship', 'love', 'family', 'father', 'mother',
        'husband', 'wife', 'brother', 'sister', 'son', 'daughter',
        'man', 'woman', 'boy', 'girl', 'character', 'story',

        # Production details
        'cinematography', 'camera', 'scene', 'shot', 'editing',
        'director', 'screenplay', 'script', 'dialogue',

        # Generic actions
        'escape', 'rescue', 'fight', 'battle', 'chase', 'running',
        'hiding', 'searching', 'looking', 'finding', 'trying'
    }

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize keyword analyzer.

        Args:
            movies_df: DataFrame with movies (must have 'genres', 'keywords', 'description')
        """
        self.movies_df = movies_df
        self.genre_keywords = {}
        self.combo_keywords = {}
        self.all_genres = self._extract_all_genres()

    def _extract_all_genres(self) -> List[str]:
        """Extract unique genres from dataset."""
        all_genres = set()
        for genres in self.movies_df['genres']:
            if isinstance(genres, (list, np.ndarray)):
                all_genres.update(genres)
        return sorted(list(all_genres))

    def build_keyword_database(self):
        """
        Build complete keyword database:
        1. Single genre keywords
        2. Genre combination keywords
        """
        print(f"Building keyword database from {len(self.movies_df):,} movies...")
        print(f"Found {len(self.all_genres)} unique genres")

        # Build single genre keywords
        print("\n1. Extracting single genre keywords...")
        for i, genre in enumerate(self.all_genres, 1):
            print(f"   [{i}/{len(self.all_genres)}] Processing '{genre}'...", end='')
            keywords = self._extract_keywords_for_genre(genre, top_k=20)
            self.genre_keywords[genre] = keywords
            print(f" {len(keywords)} keywords")

        # Build common combo keywords
        print("\n2. Extracting genre combination keywords...")
        common_combos = self._find_common_combinations()
        for i, combo in enumerate(common_combos, 1):
            print(f"   [{i}/{len(common_combos)}] Processing {combo}...", end='')
            keywords = self._extract_keywords_for_combo(combo, top_k=10)
            self.combo_keywords[tuple(sorted(combo))] = keywords
            print(f" {len(keywords)} keywords")

        print("\n✓ Keyword database built successfully!")
        print(f"  - {len(self.genre_keywords)} single genres")
        print(f"  - {len(self.combo_keywords)} genre combinations")

    def _find_common_combinations(self, min_movies: int = 100) -> List[Tuple[str, str]]:
        """
        Find common 2-genre combinations that appear in at least min_movies.

        Returns:
            List of genre pairs like [('action', 'thriller'), ...]
        """
        combo_counts = Counter()

        for genres in self.movies_df['genres']:
            if isinstance(genres, (list, np.ndarray)) and len(genres) >= 2:
                # Get all 2-genre combinations
                genre_list = list(genres)
                for i in range(len(genre_list)):
                    for j in range(i+1, len(genre_list)):
                        combo = tuple(sorted([genre_list[i], genre_list[j]]))
                        combo_counts[combo] += 1

        # Keep combos with enough movies
        common = [combo for combo, count in combo_counts.items() if count >= min_movies]
        return sorted(common, key=lambda x: combo_counts[x], reverse=True)[:50]  # Top 50

    def _extract_keywords_for_genre(self, genre: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract top keywords for a single genre using TF-IDF.

        Args:
            genre: Genre name
            top_k: Number of keywords to return

        Returns:
            List of (keyword, score) tuples, sorted by relevance
        """
        # Get movies with this genre
        genre_movies = self.movies_df[
            self.movies_df['genres'].apply(lambda g: genre in g if isinstance(g, (list, np.ndarray)) else False)
        ]

        if len(genre_movies) == 0:
            return []

        # Extract keywords from TMDB
        keyword_counts = Counter()
        for keywords in genre_movies['keywords'].dropna():
            if isinstance(keywords, (list, np.ndarray)):
                for kw in keywords:
                    kw_clean = str(kw).lower().strip()
                    if kw_clean and kw_clean not in self.GENERIC_KEYWORDS:
                        keyword_counts[kw_clean] += 1

        # Calculate TF-IDF scores
        scores = self._calculate_tfidf(keyword_counts, len(genre_movies), genre)

        # Return top K
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _extract_keywords_for_combo(self, combo: Tuple[str, str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords unique to a genre combination.

        Strategy: Find keywords that appear more often in this combo
        than in either genre alone.
        """
        genre1, genre2 = combo

        # Get movies with BOTH genres
        combo_movies = self.movies_df[
            self.movies_df['genres'].apply(
                lambda g: (genre1 in g and genre2 in g) if isinstance(g, (list, np.ndarray)) else False
            )
        ]

        if len(combo_movies) < 20:
            return []

        # Extract keywords
        keyword_counts = Counter()
        for keywords in combo_movies['keywords'].dropna():
            if isinstance(keywords, (list, np.ndarray)):
                for kw in keywords:
                    kw_clean = str(kw).lower().strip()
                    if kw_clean and kw_clean not in self.GENERIC_KEYWORDS:
                        keyword_counts[kw_clean] += 1

        # Calculate combo-specific TF-IDF
        scores = self._calculate_tfidf(keyword_counts, len(combo_movies), f"{genre1}+{genre2}")

        # Boost keywords that appear more in combo than in individual genres
        boosted_scores = {}
        for keyword, score in scores.items():
            # Check if keyword is in top keywords for individual genres
            in_genre1 = keyword in dict(self.genre_keywords.get(genre1, []))
            in_genre2 = keyword in dict(self.genre_keywords.get(genre2, []))

            # Boost combo-specific keywords
            if not in_genre1 and not in_genre2:
                boosted_scores[keyword] = score * 1.5  # 50% boost for unique keywords
            else:
                boosted_scores[keyword] = score

        return sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _calculate_tfidf(self, keyword_counts: Counter, num_docs: int, context: str) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for keywords.

        TF (Term Frequency): How often keyword appears in this genre
        IDF (Inverse Document Frequency): How unique keyword is to this genre
        """
        scores = {}

        for keyword, tf in keyword_counts.items():
            # Term frequency (normalized)
            tf_score = tf / num_docs

            # Document frequency (how many movies overall have this keyword)
            df = sum(
                1 for keywords in self.movies_df['keywords'].dropna()
                if isinstance(keywords, (list, np.ndarray)) and keyword in [str(k).lower() for k in keywords]
            )

            # IDF score
            idf_score = np.log(len(self.movies_df) / (1 + df))

            # TF-IDF
            scores[keyword] = tf_score * idf_score

        return scores

    def suggest_keywords(self, genres: List[str], num_keywords: int = 8) -> List[str]:
        """
        Suggest contextually relevant keywords based on selected genres.

        Args:
            genres: List of 1-2 selected genres
            num_keywords: Number of keywords to suggest

        Returns:
            List of keyword strings
        """
        if len(genres) == 0:
            return []

        if len(genres) == 1:
            # Single genre - return top keywords
            keywords_with_scores = self.genre_keywords.get(genres[0], [])
            return [kw for kw, score in keywords_with_scores[:num_keywords]]

        elif len(genres) == 2:
            # Two genres - blend both + combo-specific
            genre1, genre2 = genres[0], genres[1]
            combo_key = tuple(sorted(genres))

            # Get keywords from each source
            kw1 = self.genre_keywords.get(genre1, [])[:5]
            kw2 = self.genre_keywords.get(genre2, [])[:5]
            combo_kw = self.combo_keywords.get(combo_key, [])[:4]

            # Merge and deduplicate
            all_kw = kw1 + kw2 + combo_kw
            seen = set()
            result = []

            for kw, score in all_kw:
                if kw not in seen:
                    result.append(kw)
                    seen.add(kw)
                    if len(result) >= num_keywords:
                        break

            return result

        return []

    def save(self, output_path: Path):
        """Save keyword database to disk."""
        data = {
            'genre_keywords': self.genre_keywords,
            'combo_keywords': self.combo_keywords,
            'all_genres': self.all_genres
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n✓ Keyword database saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    @staticmethod
    def load(input_path: Path) -> 'KeywordAnalyzer':
        """Load keyword database from disk."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        # Create empty analyzer and populate
        analyzer = KeywordAnalyzer.__new__(KeywordAnalyzer)
        analyzer.genre_keywords = data['genre_keywords']
        analyzer.combo_keywords = data['combo_keywords']
        analyzer.all_genres = data['all_genres']
        analyzer.movies_df = None  # Not needed after loading

        return analyzer


def calculate_keyword_match_score(movie: dict, selected_keywords: List[str]) -> float:
    """
    Calculate keyword matching score for a movie.

    Args:
        movie: Movie dictionary with 'keywords' and 'description' fields
        selected_keywords: List of user-selected keywords

    Returns:
        Score between 0.0 and 1.0
    """
    if not selected_keywords:
        return 0.0

    # Extract movie keywords
    movie_keywords = movie.get('keywords', [])
    if isinstance(movie_keywords, np.ndarray):
        movie_keywords = [str(k).lower() for k in movie_keywords]
    elif isinstance(movie_keywords, list):
        movie_keywords = [str(k).lower() for k in movie_keywords]
    else:
        movie_keywords = []

    # Check TMDB keyword exact matches (70% weight)
    movie_kw_set = set(movie_keywords)
    selected_set = set(kw.lower() for kw in selected_keywords)

    exact_matches = len(movie_kw_set & selected_set)
    exact_score = exact_matches / len(selected_keywords) if selected_keywords else 0

    # Check description semantic matches (30% weight)
    description = str(movie.get('description', '')).lower()
    semantic_matches = sum(1 for kw in selected_keywords if kw.lower() in description)
    semantic_score = semantic_matches / len(selected_keywords) if selected_keywords else 0

    # Weighted combination
    final_score = 0.7 * exact_score + 0.3 * semantic_score

    return min(1.0, final_score)


if __name__ == '__main__':
    """Build keyword database from movies dataset."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python keyword_analyzer.py <movies.parquet> <output_dir>")
        sys.exit(1)

    movies_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("KEYWORD DATABASE BUILDER")
    print("="*60)

    # Load movies
    print(f"\nLoading movies from {movies_path}...")
    movies_df = pd.read_parquet(movies_path)
    print(f"✓ Loaded {len(movies_df):,} movies")

    # Build database
    analyzer = KeywordAnalyzer(movies_df)
    analyzer.build_keyword_database()

    # Save
    output_path = output_dir / 'keyword_database.pkl'
    analyzer.save(output_path)

    # Test
    print("\n" + "="*60)
    print("TEST: Keyword Suggestions")
    print("="*60)

    test_cases = [
        ['action'],
        ['comedy'],
        ['action', 'thriller'],
        ['comedy', 'romance'],
        ['horror', 'thriller']
    ]

    for genres in test_cases:
        keywords = analyzer.suggest_keywords(genres, num_keywords=8)
        print(f"\n{genres}: {keywords}")

    print("\n" + "="*60)
    print("✓ Keyword database built successfully!")
    print("="*60)
