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
# Import keyword filter for comprehensive filtering
from src.recommendation.keyword_filter import KeywordFilter


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
        """Extract unique genres from dataset, normalized (lowercase)."""
        all_genres = set()
        for genres in self.movies_df['genres']:
            if isinstance(genres, (list, np.ndarray)):
                for genre in genres:
                    # Normalize genres to lowercase for consistent matching
                    genre_normalized = str(genre).lower().strip()
                    if genre_normalized:
                        all_genres.add(genre_normalized)
        return sorted(list(all_genres))
    
    def _extract_description_terms(self, description: str, min_length: int = 3) -> List[str]:
        """
        Extract meaningful terms from movie description.
        
        Strategy:
        - Lowercase, remove punctuation
        - Extract 1-2 word n-grams (simple approach: split and filter)
        - Remove stopwords and very short terms
        
        Args:
            description: Movie description text
            min_length: Minimum term length to keep
        
        Returns:
            List of candidate terms from description
        """
        if not description or pd.isna(description):
            return []
        
        # Common stopwords to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'about'
        }
        
        # Clean description: lowercase, remove punctuation except hyphens and apostrophes
        text = str(description).lower()
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
        
        words = text.split()
        terms = []
        
        # Extract 1-word terms (meaningful words only)
        for word in words:
            word_clean = word.strip("'\"-")
            if (len(word_clean) >= min_length and 
                word_clean not in stopwords and
                not word_clean.isdigit()):
                terms.append(word_clean)
        
        # Extract 2-word terms (bigrams)
        for i in range(len(words) - 1):
            word1 = words[i].strip("'\"-")
            word2 = words[i+1].strip("'\"-")
            
            if (word1 not in stopwords and word2 not in stopwords and
                len(word1) >= min_length and len(word2) >= min_length and
                not word1.isdigit() and not word2.isdigit()):
                bigram = f"{word1} {word2}"
                # Filter out common generic bigrams
                if bigram not in {
                    'one of', 'two of', 'three of', 'four of', 'five of',
                    'part of', 'kind of', 'sort of', 'type of', 'way of',
                    'end of', 'beginning of', 'middle of', 'rest of',
                    'out of', 'up to', 'down to', 'over to', 'back to'
                }:
                    terms.append(bigram)
        
        return terms

    def build_keyword_database(self, min_movies_pair: int = 50, store_k: int = 50):
        """
        Build complete keyword database:
        1. Single genre keywords (ALL genres)
        2. Genre combination keywords (ALL pairs with enough movies)
        
        Args:
            min_movies_pair: Minimum movies required for a pair (default 50, lower for completeness)
            store_k: Number of keywords to store per context (default 50, API returns top 8)
        """
        print(f"Building keyword database from {len(self.movies_df):,} movies...")
        print(f"Found {len(self.all_genres)} unique genres")
        print(f"Using min_movies_pair={min_movies_pair}, storing top {store_k} keywords per context")

        # Build single genre keywords (ALL genres)
        print(f"\n1. Extracting single genre keywords for ALL {len(self.all_genres)} genres...")
        for i, genre in enumerate(self.all_genres, 1):
            print(f"   [{i}/{len(self.all_genres)}] Processing '{genre}'...", end='', flush=True)
            keywords = self._extract_keywords_for_genre(genre, top_k=store_k)
            self.genre_keywords[genre] = keywords
            print(f" {len(keywords)} keywords")

        # Build combo keywords (ALL pairs with enough movies, not just top 50)
        print(f"\n2. Finding ALL genre combinations (min {min_movies_pair} movies)...")
        common_combos = self._find_common_combinations(min_movies=min_movies_pair)
        print(f"   Found {len(common_combos)} genre pairs meeting threshold")
        
        print(f"\n3. Extracting keywords for ALL {len(common_combos)} genre pairs...")
        for i, combo in enumerate(common_combos, 1):
            genre1, genre2 = combo
            print(f"   [{i}/{len(common_combos)}] Processing {genre1}+{genre2}...", end='', flush=True)
            keywords = self._extract_keywords_for_combo(combo, top_k=store_k)
            # Store with normalized sorted key
            combo_key = tuple(sorted([str(c).lower().strip() for c in combo]))
            self.combo_keywords[combo_key] = keywords
            print(f" {len(keywords)} keywords")

        print("\n✓ Keyword database built successfully!")
        print(f"  - {len(self.genre_keywords)} single genres")
        print(f"  - {len(self.combo_keywords)} genre combinations (ALL pairs, not just top 50)")

    def _find_common_combinations(self, min_movies: int = 50) -> List[Tuple[str, str]]:
        """
        Find ALL 2-genre combinations that appear in at least min_movies.
        
        IMPORTANT: Returns ALL pairs meeting threshold (not just top 50) to ensure
        comprehensive keyword coverage for all genre combinations.

        Args:
            min_movies: Minimum number of movies with this pair (default 50, lower for completeness)

        Returns:
            List of genre pairs like [('action', 'thriller'), ...] - ALL pairs, not just top 50
        """
        combo_counts = Counter()

        for genres in self.movies_df['genres']:
            if isinstance(genres, (list, np.ndarray)) and len(genres) >= 2:
                # Get all 2-genre combinations, normalized to lowercase
                genre_list = [str(g).lower().strip() for g in genres]
                for i in range(len(genre_list)):
                    for j in range(i+1, len(genre_list)):
                        # Normalize both genres and sort for consistent keys
                        combo = tuple(sorted([genre_list[i], genre_list[j]]))
                        combo_counts[combo] += 1

        # Keep ALL combos with enough movies (not just top 50)
        common = [combo for combo, count in combo_counts.items() if count >= min_movies]
        # Sort by count for processing priority (most common first)
        return sorted(common, key=lambda x: combo_counts[x], reverse=True)

    def _extract_keywords_for_genre(self, genre: str, top_k: int = 50, min_rating: float = 6.5) -> List[Tuple[str, float]]:
        """
        Extract top keywords for a single genre using TF-IDF with quality focus.
        
        Sources:
        - TMDb keywords (from keywords column)
        - Description-derived terms (from description column)
        
        Args:
            genre: Genre name (normalized, lowercase)
            top_k: Number of keywords to return (default 50 for storage, API returns 8)
            min_rating: Minimum rating to focus on quality movies (default 6.5, more lenient)

        Returns:
            List of (keyword, score) tuples, sorted by relevance
        """
        # Normalize genre for matching
        genre_normalized = genre.lower().strip()
        
        # Get movies with this genre (case-insensitive matching)
        genre_movies = self.movies_df[
            self.movies_df['genres'].apply(
                lambda g: genre_normalized in [str(gen).lower().strip() for gen in g] 
                if isinstance(g, (list, np.ndarray)) else False
            )
        ]

        if len(genre_movies) == 0:
            return []

        # Focus on quality movies for keyword extraction (more lenient threshold)
        quality_movies = genre_movies[genre_movies['avg_rating'] >= min_rating]

        if len(quality_movies) < 20:
            # Fallback to all genre movies if not enough high-rated
            quality_movies = genre_movies

        # Extract keywords from TMDb keywords column
        keyword_counts = Counter()
        for idx, row in quality_movies.iterrows():
            # TMDb keywords
            keywords = row.get('keywords', [])
            if isinstance(keywords, (list, np.ndarray)):
                for kw in keywords:
                    kw_clean = str(kw).lower().strip()
                    if kw_clean and KeywordFilter.is_relevant(kw_clean):
                        keyword_counts[kw_clean] += 1
            
            # Description-derived terms
            description = row.get('description', '')
            if description:
                desc_terms = self._extract_description_terms(str(description))
                for term in desc_terms:
                    term_clean = term.lower().strip()
                    if term_clean and KeywordFilter.is_relevant(term_clean):
                        # Description terms get slightly lower weight (0.8x) vs TMDb keywords
                        keyword_counts[term_clean] += 0.8

        if not keyword_counts:
            return []

        # Calculate TF-IDF scores with quality boost
        scores = self._calculate_tfidf_with_quality(
            keyword_counts,
            len(quality_movies),
            genre_normalized,
            quality_movies
        )

        # Apply quality filters (frequency constraints)
        filtered_scores = self._apply_quality_filters(scores, genre_movies, [genre_normalized])

        # Return top K (50 for storage, API will return top 8)
        return sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _extract_keywords_for_combo(self, combo: Tuple[str, str], top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Extract keywords for a genre combination.
        
        Strategy (improved):
        1. Extract keywords from movies with BOTH genres
        2. Prefer pair-specific keywords (those distinctive to the combo)
        3. But also include shared/intersection keywords if pair-specific is sparse
        4. Sources: TMDb keywords + description terms
        
        Args:
            combo: Tuple of (genre1, genre2) - both normalized lowercase
            top_k: Number of keywords to return (default 50 for storage)
        
        Returns:
            List of (keyword, score) tuples, sorted by relevance
        """
        genre1, genre2 = combo
        genre1_normalized = genre1.lower().strip()
        genre2_normalized = genre2.lower().strip()

        # Get movies with BOTH genres (case-insensitive)
        combo_movies = self.movies_df[
            self.movies_df['genres'].apply(
                lambda g: (
                    genre1_normalized in [str(gen).lower().strip() for gen in g] and
                    genre2_normalized in [str(gen).lower().strip() for gen in g]
                ) if isinstance(g, (list, np.ndarray)) else False
            )
        ]

        if len(combo_movies) < 10:  # Lower threshold for pairs (min 10 instead of 20)
            return []

        # Extract keywords from TMDb + descriptions
        keyword_counts = Counter()
        for idx, row in combo_movies.iterrows():
            # TMDb keywords
            keywords = row.get('keywords', [])
            if isinstance(keywords, (list, np.ndarray)):
                for kw in keywords:
                    kw_clean = str(kw).lower().strip()
                    if kw_clean and KeywordFilter.is_relevant(kw_clean):
                        keyword_counts[kw_clean] += 1
            
            # Description-derived terms
            description = row.get('description', '')
            if description:
                desc_terms = self._extract_description_terms(str(description))
                for term in desc_terms:
                    term_clean = term.lower().strip()
                    if term_clean and KeywordFilter.is_relevant(term_clean):
                        keyword_counts[term_clean] += 0.8

        if not keyword_counts:
            return []

        # Calculate combo-specific TF-IDF
        scores = self._calculate_tfidf(keyword_counts, len(combo_movies), f"{genre1_normalized}+{genre2_normalized}")

        # Apply frequency filters
        filtered_scores = self._apply_quality_filters(scores, combo_movies, [genre1_normalized, genre2_normalized])

        # Strategy: Prefer pair-specific keywords but don't exclude all shared ones
        # Get top keywords from individual genres (for reference)
        genre1_top_keywords = set([kw for kw, _ in self.genre_keywords.get(genre1_normalized, [])[:20]])
        genre2_top_keywords = set([kw for kw, _ in self.genre_keywords.get(genre2_normalized, [])[:20]])
        
        # Separate pair-specific vs shared keywords
        pair_specific = {}
        shared_keywords = {}
        
        for keyword, score in filtered_scores.items():
            # Keywords NOT in top-20 of either individual genre are "pair-specific"
            if keyword not in genre1_top_keywords and keyword not in genre2_top_keywords:
                pair_specific[keyword] = score
            else:
                # Shared keywords get slightly lower weight
                shared_keywords[keyword] = score * 0.7
        
        # Combine: 70% pair-specific, 30% shared (if pair-specific is sparse, use more shared)
        if len(pair_specific) >= top_k // 2:
            # Enough pair-specific: use mostly those
            combined = dict(list(sorted(pair_specific.items(), key=lambda x: x[1], reverse=True)[:int(top_k * 0.7)]))
            # Add best shared keywords
            for kw, score in sorted(shared_keywords.items(), key=lambda x: x[1], reverse=True)[:int(top_k * 0.3)]:
                if kw not in combined:
                    combined[kw] = score
        else:
            # Sparse pair-specific: use what we have + more shared
            combined = dict(pair_specific)  # All pair-specific
            # Add shared keywords to fill up to top_k
            shared_count = min(top_k - len(combined), len(shared_keywords))
            for kw, score in sorted(shared_keywords.items(), key=lambda x: x[1], reverse=True)[:shared_count]:
                combined[kw] = score

        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

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
        context_size = len(genre_movies)

        for keyword, score in scores.items():
            keyword_lower = keyword.lower().strip()
            
            # 1. Use KeywordFilter for comprehensive thematic filtering
            if not KeywordFilter.is_relevant(keyword_lower):
                continue
            
            # 2. Check if keyword is genre synonym (exclude)
            is_synonym = False
            for genre in genres:
                genre_lower = str(genre).lower().strip()
                if genre_lower in GENRE_SYNONYMS:
                    if keyword_lower in GENRE_SYNONYMS[genre_lower]:
                        is_synonym = True
                        break
                # Also check if keyword contains genre name (exclude "action movie", etc.)
                if genre_lower in keyword_lower or keyword_lower in genre_lower:
                    # Allow exceptions like "action comedy" (compound genres are OK)
                    if ' ' in keyword_lower and keyword_lower.count(' ') == 1:
                        # Single-word genre matches are synonyms
                        parts = keyword_lower.split()
                        if genre_lower == parts[0] or genre_lower == parts[-1]:
                            is_synonym = True
                            break
                    elif genre_lower == keyword_lower:
                        is_synonym = True
                        break

            if is_synonym:
                continue

            # 3. Check frequency in context (appears in 3-40% of context movies)
            # This ensures keywords are neither too rare (noise) nor too common (generic)
            movies_with_kw = 0
            
            # Check in keywords column and description
            for idx, row in genre_movies.iterrows():
                keywords = row.get('keywords', [])
                if isinstance(keywords, (list, np.ndarray)):
                    if keyword_lower in [str(k).lower().strip() for k in keywords]:
                        movies_with_kw += 1
                        continue
                
                # Also check in description
                description = row.get('description', '')
                if description and keyword_lower in str(description).lower():
                    movies_with_kw += 1

            frequency = movies_with_kw / context_size if context_size > 0 else 0
            
            # Frequency constraints with adaptive thresholds
            min_frequency = max(0.03, 3 / context_size)  # At least 3 movies or 3%
            max_frequency = 0.40  # At most 40% of context
            
            if min_frequency <= frequency <= max_frequency:
                # Perfect frequency range
                filtered[keyword] = score
            elif frequency < min_frequency and score > 0.6:
                # Very rare but high-scoring (highly distinctive): allow with penalty
                filtered[keyword] = score * 0.7
            elif frequency > max_frequency and score > 0.4:
                # Common but still somewhat distinctive: allow with penalty
                filtered[keyword] = score * 0.6
            # Otherwise exclude (too rare with low score, or too common)

        return filtered

    def suggest_keywords(self, genres: List[str], num_keywords: int = 8) -> List[str]:
        """
        Suggest contextually relevant keywords based on selected genres.
        
        Strategy:
        - Single genre: Return top keywords for that genre
        - Two genres: Return pair-specific keywords if available, with fallback
          to merged single-genre keywords if pair is sparse or not found
        
        Args:
            genres: List of 1-2 selected genres (normalized lowercase expected)
            num_keywords: Number of keywords to suggest (default 8)

        Returns:
            List of keyword strings (always tries to return num_keywords, uses fallback if needed)
        """
        if len(genres) == 0:
            return []

        # Normalize genres
        genres_normalized = [str(g).lower().strip() for g in genres if g]

        if len(genres_normalized) == 1:
            # Single genre - return top keywords
            genre = genres_normalized[0]
            keywords_with_scores = self.genre_keywords.get(genre, [])
            keywords = [kw for kw, score in keywords_with_scores[:num_keywords]]
            
            # If still empty, try to find genre with slight variations
            if not keywords:
                # Try to find matching genre (e.g., "sci-fi" vs "science fiction")
                for stored_genre in self.genre_keywords.keys():
                    if genre in stored_genre or stored_genre in genre:
                        keywords_with_scores = self.genre_keywords.get(stored_genre, [])
                        keywords = [kw for kw, score in keywords_with_scores[:num_keywords]]
                        break
            
            return keywords

        elif len(genres_normalized) == 2:
            # Two genres - try pair-specific first, fallback to merged singles
            combo_key = tuple(sorted(genres_normalized))
            combo_kw = self.combo_keywords.get(combo_key, [])
            
            if combo_kw:
                # We have pair-specific keywords: use them
                keywords = [kw for kw, score in combo_kw[:num_keywords]]
                if len(keywords) >= num_keywords:
                    return keywords
                # If we got some but not enough, fill with fallback
            
            # Fallback: merge single-genre keywords (intersection + top from each)
            genre1, genre2 = genres_normalized
            genre1_kw = self.genre_keywords.get(genre1, [])
            genre2_kw = self.genre_keywords.get(genre2, [])
            
            if not genre1_kw and not genre2_kw:
                return []  # No data at all
            
            # Strategy: 50% intersection (keywords in both), 50% best from each genre
            genre1_keywords = {kw for kw, score in genre1_kw[:30]}  # Top 30 from genre1
            genre2_keywords = {kw for kw, score in genre2_kw[:30]}  # Top 30 from genre2
            
            # Intersection keywords (common to both genres)
            intersection = genre1_keywords & genre2_keywords
            intersection_scores = {}
            for kw, score in genre1_kw:
                if kw in intersection:
                    intersection_scores[kw] = score
            for kw, score in genre2_kw:
                if kw in intersection:
                    intersection_scores[kw] = max(intersection_scores.get(kw, 0), score)
            
            # Unique keywords from each genre (not in intersection)
            unique_scores = {}
            for kw, score in genre1_kw:
                if kw not in intersection:
                    unique_scores[kw] = max(unique_scores.get(kw, 0), score * 0.8)  # Slight penalty for uniqueness
            for kw, score in genre2_kw:
                if kw not in intersection:
                    unique_scores[kw] = max(unique_scores.get(kw, 0), score * 0.8)
            
            # Combine: 50% intersection (if available), 50% unique best
            combined = []
            
            # Add intersection keywords (up to num_keywords // 2)
            intersection_sorted = sorted(intersection_scores.items(), key=lambda x: x[1], reverse=True)
            combined.extend([kw for kw, score in intersection_sorted[:num_keywords // 2]])
            
            # Add unique keywords to fill up
            unique_sorted = sorted(unique_scores.items(), key=lambda x: x[1], reverse=True)
            remaining = num_keywords - len(combined)
            for kw, score in unique_sorted:
                if kw not in combined and len(combined) < num_keywords:
                    combined.append(kw)
            
            # If we had some pair-specific keywords, merge them in (prioritize)
            if combo_kw:
                pair_keywords = [kw for kw, score in combo_kw]
                # Prepend pair-specific, remove duplicates, limit to num_keywords
                final = []
                for kw in pair_keywords + combined:
                    if kw not in final:
                        final.append(kw)
                        if len(final) >= num_keywords:
                            break
                return final
            
            return combined[:num_keywords]

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

    # Build database with improved parameters
    analyzer = KeywordAnalyzer(movies_df)
    analyzer.build_keyword_database(
        min_movies_pair=50,  # Lower threshold to build ALL pairs (not just top 50)
        store_k=50  # Store top 50 keywords per context (API returns top 8)
    )

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
        ['sci-fi'],  # Test sci-fi (common missing case)
        ['sci-fi', 'horror'],  # Test pair with sci-fi
        ['action', 'thriller'],
        ['comedy', 'romance'],
        ['horror', 'thriller'],
        ['mystery', 'fantasy']  # Test another pair
    ]

    print("\nTesting keyword suggestions for various genres/pairs:\n")
    for genres in test_cases:
        keywords = analyzer.suggest_keywords(genres, num_keywords=8)
        status = "✓" if keywords else "✗ (EMPTY)"
        print(f"{status} {genres}: {keywords[:8] if keywords else 'No keywords found'}")

    print("\n" + "="*60)
    print("✓ Keyword database built successfully!")
    print("="*60)
