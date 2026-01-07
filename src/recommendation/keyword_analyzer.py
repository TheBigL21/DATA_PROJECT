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
        'based on novel or book', 'based on play or musical', 'based on true story',
        'remake', 'sequel', 'prequel', 'reboot', 'spin off',
        'black and white', 'color', 'silent film', 'independent film',
        'duringcreditsstinger', 'aftercreditsstinger', 'post credits scene',
        'technicolor', 'cinemascope', 'pre-code', 'b movie',

        # Generic descriptors
        'low budget', 'cult film', 'cult classic', 'surprise ending',
        'flashback', 'voice over narration', 'narration',
        'art film', 'arthouse', 'critically acclaimed', 'controversial', 'banned film',

        # Too broad/vague
        'male protagonist', 'female protagonist', 'protagonist',
        'good versus evil', 'violence', 'death', 'murder',
        'friendship', 'love', 'family', 'father', 'mother',
        'husband', 'wife', 'brother', 'sister', 'son', 'daughter',
        'man', 'woman', 'boy', 'girl', 'character', 'story',
        'husband wife relationship', 'parent child relationship',

        # Production details
        'cinematography', 'camera', 'scene', 'shot', 'editing',
        'director', 'screenplay', 'script', 'dialogue',

        # Generic actions
        'escape', 'rescue', 'fight', 'battle', 'chase', 'running',
        'hiding', 'searching', 'looking', 'finding', 'trying',

        # Identity/representation
        'lgbt', 'gay', 'lesbian', 'transgender', 'bisexual', 'queer', 'lgbtq',
        'gay theme', 'lesbian relationship', 'gay relationship',

        # Demographic
        'african american', 'black people', 'hispanic', 'latino', 'latina',
        'asian', 'white people', 'race relations', 'racial issues',

        # Age/life stage (too vague)
        'coming of age', 'midlife crisis', 'teenage', 'childhood', 'elderly',
        'adolescence', 'youth', 'old age', 'growing up',

        # Relationship dynamics (too generic)
        'interracial relationship', 'age difference', 'class differences',
        'forbidden love', 'romance', 'breakup', 'divorce', 'infidelity',
        'love triangle', 'unrequited love', 'extramarital affair',
        'marriage', 'wedding', 'engagement',

        # Standalone content descriptors
        'nudity', 'sex', 'sexuality', 'sexual content', 'erotic', 'sensuality',
        'strong language', 'profanity', 'gore', 'graphic violence',
        'sex scene', 'sexual abuse', 'rape', 'sexual violence',

        # Generic emotions/states
        'jealousy', 'revenge', 'betrayal', 'loss', 'grief', 'trauma',
        'fear', 'hope', 'despair', 'guilt', 'redemption',

        # Generic locations (too broad)
        'small town', 'big city', 'new york city', 'los angeles california',
        'paris, france', 'london, england', 'san francisco california',

        # Time periods (too generic without context)
        '19th century', '18th century', '17th century', '16th century',
        '15th century', '1st century', '1900s', '1910s', '1940s',

        # Historical (too broad)
        'world war ii', 'world war i', 'historical figure', 'biography',
        'based on true story',

        # Film noir elements (standalone)
        'film noir', 'british noir', 'western noir',

        # Generic plot elements
        'trial', 'investigation', 'murder investigation', 'murder mystery',
        'on the run', 'fugitive', 'kidnapping', 'blackmail', 'deception',
        'mistaken identity', 'assumed identity', 'framed for murder'
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

    def _find_common_combinations(self, min_movies: int = 200) -> List[Tuple[str, str]]:
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

    def _extract_keywords_for_genre(self, genre: str, top_k: int = 20, min_rating: float = 7.0) -> List[Tuple[str, float]]:
        """
        Extract top keywords for a single genre using TF-IDF with quality focus.

        Args:
            genre: Genre name
            top_k: Number of keywords to return
            min_rating: Minimum rating to focus on quality movies (default 7.0)

        Returns:
            List of (keyword, score) tuples, sorted by relevance
        """
        # Get movies with this genre
        genre_movies = self.movies_df[
            self.movies_df['genres'].apply(lambda g: genre in g if isinstance(g, (list, np.ndarray)) else False)
        ]

        if len(genre_movies) == 0:
            return []

        # Focus on quality movies for keyword extraction
        quality_movies = genre_movies[genre_movies['avg_rating'] >= min_rating]

        if len(quality_movies) < 20:
            # Fallback to lower threshold if not enough high-rated movies
            quality_movies = genre_movies[genre_movies['avg_rating'] >= 6.5]

        # Extract keywords from TMDB
        keyword_counts = Counter()
        for keywords in quality_movies['keywords'].dropna():
            if isinstance(keywords, (list, np.ndarray)):
                for kw in keywords:
                    kw_clean = str(kw).lower().strip()
                    if kw_clean and kw_clean not in self.GENERIC_KEYWORDS:
                        keyword_counts[kw_clean] += 1

        # Calculate TF-IDF scores with quality boost
        scores = self._calculate_tfidf_with_quality(
            keyword_counts,
            len(quality_movies),
            genre,
            quality_movies
        )

        # Apply quality filters
        filtered_scores = self._apply_quality_filters(scores, genre_movies, [genre])

        # Return top K
        return sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _extract_keywords_for_combo(self, combo: Tuple[str, str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords unique to a genre combination.

        Strategy: Find keywords that appear in this combo but NOT in
        the top keywords of either individual genre.
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

        # HARD EXCLUDE keywords that appear in top-15 of either individual genre
        genre1_top_keywords = set([kw for kw, _ in self.genre_keywords.get(genre1, [])[:15]])
        genre2_top_keywords = set([kw for kw, _ in self.genre_keywords.get(genre2, [])[:15]])

        filtered_scores = {}
        for keyword, score in scores.items():
            # Only keep keywords NOT common in either individual genre
            if keyword not in genre1_top_keywords and keyword not in genre2_top_keywords:
                filtered_scores[keyword] = score

        return sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

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

    def _calculate_tfidf_with_quality(self, keyword_counts: Counter, num_docs: int,
                                      context: str, quality_movies: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate TF-IDF scores with quality boost for keywords.

        Keywords appearing in higher-rated movies get a boost.
        """
        scores = {}

        for keyword, tf in keyword_counts.items():
            # Base TF-IDF
            tf_score = tf / num_docs

            # Document frequency
            df = sum(
                1 for keywords in self.movies_df['keywords'].dropna()
                if isinstance(keywords, (list, np.ndarray)) and keyword in [str(k).lower() for k in keywords]
            )

            idf_score = np.log(len(self.movies_df) / (1 + df))

            # Quality boost: calculate average rating of movies with this keyword
            movies_with_kw = quality_movies[
                quality_movies['keywords'].apply(
                    lambda kws: keyword in [str(k).lower() for k in kws] if isinstance(kws, (list, np.ndarray)) else False
                )
            ]

            if len(movies_with_kw) > 0:
                avg_rating = movies_with_kw['avg_rating'].mean()
                # Boost: 7.0 rating = 1.0x, 8.0 rating = 1.2x, 8.5 rating = 1.3x
                quality_boost = 1.0 + max(0, (avg_rating - 7.0) * 0.2)
            else:
                quality_boost = 1.0

            # Final score
            scores[keyword] = tf_score * idf_score * quality_boost

        return scores

    def _apply_quality_filters(self, scores: Dict[str, float],
                               genre_movies: pd.DataFrame,
                               genres: List[str]) -> Dict[str, float]:
        """
        Apply quality filters to keyword scores:
        1. Specificity check (not too common)
        2. Remove genre synonyms
        """
        # Genre synonyms to exclude
        GENRE_SYNONYMS = {
            'action': ['action hero', 'action packed', 'explosive action', 'high octane'],
            'thriller': ['suspense', 'suspenseful', 'thrilling', 'tension'],
            'comedy': ['funny', 'humor', 'humorous', 'laugh', 'hilarious'],
            'horror': ['scary', 'terrifying', 'frightening', 'spooky'],
            'drama': ['dramatic', 'emotional', 'moving'],
            'romance': ['romantic', 'love story'],
            'sci-fi': ['futuristic', 'science fiction'],
            'fantasy': ['magical', 'fantastical'],
        }

        filtered = {}

        for keyword, score in scores.items():
            # Check if keyword is genre synonym
            is_synonym = False
            for genre in genres:
                genre_lower = genre.lower()
                if genre_lower in GENRE_SYNONYMS:
                    if keyword in GENRE_SYNONYMS[genre_lower]:
                        is_synonym = True
                        break
                # Also check if keyword contains genre name
                if genre_lower in keyword or keyword in genre_lower:
                    is_synonym = True
                    break

            if is_synonym:
                continue

            # Check specificity (appears in 5-40% of genre movies)
            movies_with_kw = genre_movies[
                genre_movies['keywords'].apply(
                    lambda kws: keyword in [str(k).lower() for k in kws] if isinstance(kws, (list, np.ndarray)) else False
                )
            ]

            frequency = len(movies_with_kw) / len(genre_movies)

            # Keep keywords with reasonable frequency
            if 0.05 <= frequency <= 0.40:
                filtered[keyword] = score
            elif frequency < 0.05 and score > 0.5:
                # Rare but high-scoring keywords are OK
                filtered[keyword] = score * 0.7  # Slightly penalize

        return filtered

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
            # Two genres - return ONLY combo-specific keywords
            combo_key = tuple(sorted(genres))
            combo_kw = self.combo_keywords.get(combo_key, [])
            return [kw for kw, score in combo_kw[:num_keywords]]

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
