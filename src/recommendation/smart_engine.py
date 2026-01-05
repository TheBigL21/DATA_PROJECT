"""
SMART RECOMMENDATION ENGINE (Option C)

New recommendation system with:
1. Option C quality scoring (context-aware, probabilistic)
2. Simplified user flow (evening type → genres → era)
3. Smart keyword matching
4. Era favorability (not filtering)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommendation.recommendation_engine import RecommendationEngine
from recommendation.keyword_analyzer import KeywordAnalyzer, calculate_keyword_match_score
from recommendation.inference_engine import (
    build_user_model,
    calculate_popularity_boost,
    infer_pacing_from_session
)
from recommendation.time_period_filter import TimePeriodFilter
from recommendation.source_material_filter import SourceMaterialFilter
from recommendation.age_keywords import calculate_age_score
from recommendation.pacing_calculator import calculate_pacing_match


class SmartRecommendationEngine:
    """
    Smart recommendation engine with Option C quality scoring and keyword matching.
    """

    # Vote confidence constants
    MAX_VOTES_NORMALIZATION = 2_000_000  # Reference point for vote confidence
    MIN_CONFIDENCE = 0.3  # Minimum confidence multiplier
    MAX_CONFIDENCE = 1.0  # Maximum confidence multiplier

    # Era configurations now use TimePeriodFilter (no local config needed)

    # Evening type modifiers for quality scoring
    EVENING_MODIFIERS = {
        'date_night': 1.2,      # Want impressive movies
        'family_night': 1.1,    # Want reliable entertainment
        'friends_night': 1.0,   # Social focus, forgiving
        'chill_evening': 0.9    # Experimental mood
    }

    def __init__(
        self,
        base_engine: RecommendationEngine,
        keyword_analyzer: Optional[KeywordAnalyzer] = None
    ):
        """
        Initialize smart recommendation engine.

        Args:
            base_engine: Base engine with CF and graph models
            keyword_analyzer: Optional keyword analyzer for suggestions
        """
        self.base_engine = base_engine
        self.movies = base_engine.movies
        self.keyword_analyzer = keyword_analyzer

        # Base scoring weights (when NO keywords selected)
        self.w_genre_base = 0.30
        self.w_cf_base = 0.30
        self.w_graph_base = 0.05
        self.w_quality_base = 0.20
        self.w_session_base = 0.15

        # Scoring weights with keywords
        self.w_genre_kw = 0.25      # Genre matching (primary user intent)
        self.w_keywords = 0.30      # INCREASED - keyword matching is now primary
        self.w_cf_kw = 0.20         # Collaborative filtering
        self.w_quality_kw = 0.15    # Quality score (dynamically adjusted)
        self.w_session_kw = 0.10    # Session similarity

    def _calculate_vote_confidence(self, num_votes: int) -> float:
        """
        Calculate confidence multiplier from vote count.

        Uses logarithmic scaling to reward movies with more votes.
        Clamps result between MIN_CONFIDENCE and MAX_CONFIDENCE.

        Args:
            num_votes: Number of votes for the movie

        Returns:
            Confidence multiplier in range [0.3, 1.0]
        """
        confidence = np.log1p(num_votes) / np.log1p(self.MAX_VOTES_NORMALIZATION)
        return max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, confidence))

    def recommend(
        self,
        user_id: int,
        genres: List[str],
        era: Optional[str] = None,
        source_material: Optional[str] = None,
        themes: Optional[List[str]] = None,
        session_history: Optional[List[Dict]] = None,
        top_k: int = 20
    ) -> List[int]:
        """
        Generate recommendations with new filtering system.

        Args:
            user_id: User ID
            genres: List of 1-2 selected genres
            era: Selected time period ('new_era', 'millennium', 'old_school', 'golden_era', 'any')
            source_material: Source material preference ('book', 'true_story', 'any', etc.)
            themes: List of thematic keywords selected by user
            session_history: List of {movie_id, action, num_votes, pacing_score} dicts
            top_k: Number of recommendations to return

        Returns:
            List of movie_ids
        """
        if session_history is None:
            session_history = []
        if themes is None:
            themes = []
        if era is None:
            era = 'any'
        if source_material is None:
            source_material = 'any'

        # Extract session context
        session_positive = [
            item['movie_id'] for item in session_history
            if item.get('action') in ['right', 'up']
        ]
        already_shown = [item['movie_id'] for item in session_history]

        # STAGE 1: Generate candidates
        candidates = self._generate_candidates(
            user_id=user_id,
            genres=genres,
            era=era,
            quality_floor=6.0,  # Quality floor set to 6.0
            session_positive_movies=session_positive,
            already_shown=already_shown
        )

        # STAGE 2: Score all candidates (strict scoring)
        scores = self._score_candidates_new(
            candidates=candidates,
            user_id=user_id,
            genres=genres,
            era=era,
            source_material=source_material,
            themes=themes,
            session_positive_movies=session_positive,
            session_history=session_history
        )

        # Store strict scores for interactive selector (80/20 split)
        self.strict_scores = dict(zip(candidates['movie_id'].tolist(), scores))

        # STAGE 3: Sort and return top K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return candidates.iloc[top_indices]['movie_id'].tolist()

    def get_strict_scores(self) -> Dict[int, float]:
        """
        Get strict scores for all candidates (used by InteractiveSelector).

        Returns:
            Dict mapping movie_id → strict score [0.0, 1.0]
        """
        return getattr(self, 'strict_scores', {})

    def _generate_candidates(
        self,
        user_id: int,
        genres: List[str],
        era: str,
        quality_floor: float,
        session_positive_movies: List[int],
        already_shown: List[int]
    ) -> pd.DataFrame:
        """
        Generate candidate movies using HARD filtering (strict requirements).

        HARD FILTERS (non-negotiable):
        - Genre: EXACT match required (at least one selected genre)
        - Quality: rating >= quality_floor (6.0)
        - Popularity: num_votes >= 5000
        - Era: Movies must be within exact year range (if era != 'any')
          (uses TimePeriodFilter for consistent era definitions)

        Returns 50-200 high-quality candidates.
        """
        # HARD FILTER 1: Genre match (case-insensitive)
        genres_lower = [g.lower().strip() for g in genres]
        def matches_genre(movie_genres):
            if not isinstance(movie_genres, (list, np.ndarray)):
                return False
            movie_genres_lower = [str(g).lower().strip() for g in movie_genres]
            return any(genre_lower in movie_genres_lower for genre_lower in genres_lower)
        
        genre_mask = self.movies['genres'].apply(matches_genre)

        # HARD FILTER 2: Quality floor (6.0 minimum)
        rating_mask = self.movies['avg_rating'] >= quality_floor

        # HARD FILTER 3: Popularity floor (5000 votes minimum)
        popularity_mask = self.movies['num_votes'] >= 5000

        # HARD FILTER 4: Era range (exact year match required)
        if era != 'any':
            year_min, year_max = TimePeriodFilter.get_year_range(era)
            if year_min is not None and year_max is not None:
                era_mask = (self.movies['year'] >= year_min) & (self.movies['year'] <= year_max)
            else:
                era_mask = pd.Series([True] * len(self.movies), index=self.movies.index)
        else:
            era_mask = pd.Series([True] * len(self.movies), index=self.movies.index)

        # Apply all hard filters
        candidates = self.movies[genre_mask & rating_mask & popularity_mask & era_mask].copy()

        # Remove already shown
        candidates = candidates[~candidates['movie_id'].isin(already_shown)]

        # Limit to top 200 by votes (popularity proxy for quality)
        if len(candidates) > 200:
            candidates = candidates.nlargest(200, 'num_votes')

        return candidates

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        genres: List[str],
        selected_keywords: Dict[str, Any],
        user_model: Dict[str, Any],
        session_positive_movies: List[int],
        session_history: List[Dict]
    ) -> np.ndarray:
        """
        Score all candidates using weighted components with inference.

        Returns:
            Array of scores (one per candidate)
        """
        n_candidates = len(candidates)

        # Determine if keywords were selected
        has_keywords = (
            selected_keywords.get('age') or
            selected_keywords.get('general') or
            selected_keywords.get('adaptation') or
            selected_keywords.get('pacing')
        )

        # Select weights based on keyword usage
        if has_keywords:
            w_genre = self.w_genre_kw
            w_keywords = self.w_keywords
            w_cf = self.w_cf_kw
            w_quality = user_model['quality']['quality_weight']  # Dynamic
            w_session = self.w_session_kw + (0.05 if len(session_history) >= 5 else 0)
            w_graph = 0.05
        else:
            w_genre = self.w_genre_base
            w_keywords = 0.0
            w_cf = self.w_cf_base
            w_quality = self.w_quality_base
            w_session = self.w_session_base
            w_graph = self.w_graph_base

        # Normalize weights
        total_w = w_genre + w_keywords + w_cf + w_quality + w_session + w_graph
        w_genre /= total_w
        w_keywords /= total_w
        w_cf /= total_w
        w_quality /= total_w
        w_session /= total_w
        w_graph /= total_w

        # Calculate component scores
        scores = np.zeros(n_candidates)

        # Component 1: Genre matching
        genre_scores = self._get_genre_scores(candidates, genres)
        scores += w_genre * genre_scores

        # Component 2: Keyword matching (if keywords selected)
        if has_keywords:
            keyword_scores = self._get_keyword_scores_enhanced(
                candidates,
                selected_keywords,
                user_model['pacing']
            )
            scores += w_keywords * keyword_scores

        # Component 3: Collaborative Filtering
        cf_scores = self._get_cf_scores(candidates, user_id)
        scores += w_cf * cf_scores

        # Component 4: Quality score
        quality_scores = self._get_quality_scores_simple(candidates)
        scores += w_quality * quality_scores

        # Component 5: Session similarity (with pacing awareness)
        session_scores = self._get_session_scores_enhanced(
            candidates,
            genres,
            session_positive_movies,
            session_history,
            user_model['pacing']
        )
        scores += w_session * session_scores

        # Component 6: Graph similarity
        graph_scores = self._get_graph_scores(candidates, session_positive_movies)
        scores += w_graph * graph_scores

        return scores

    def _get_genre_scores(self, candidates: pd.DataFrame, selected_genres: List[str]) -> np.ndarray:
        """
        Calculate explicit genre matching scores (0-1 normalized).

        Scoring:
        - Both selected genres match: 1.0
        - One selected genre matches: 0.5
        - No match: 0.0 (but already filtered out in candidate generation)

        Args:
            candidates: DataFrame of candidate movies
            selected_genres: List of 1-2 genres selected by user

        Returns:
            Array of genre match scores (0-1)
        """
        scores = np.zeros(len(candidates))
        selected_set = {g.lower() for g in selected_genres}

        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_genres = {g.lower() for g in row['genres']} if isinstance(row['genres'], (list, np.ndarray)) else set()

            matches = len(movie_genres & selected_set)

            if len(selected_genres) == 1:
                # Single genre: binary match
                scores[idx] = 1.0 if matches > 0 else 0.0
            else:
                # Two genres: 1.0 for both, 0.5 for one, 0.0 for none
                if matches >= 2:
                    scores[idx] = 1.0
                elif matches == 1:
                    scores[idx] = 0.5
                else:
                    scores[idx] = 0.0

        return scores

    def _get_cf_scores(self, candidates: pd.DataFrame, user_id: int) -> np.ndarray:
        """Get collaborative filtering scores (0-1 normalized)."""
        if not hasattr(self.base_engine, 'cf_model') or not self.base_engine.cf_model:
            # No CF model, return neutral
            return np.ones(len(candidates)) * 0.5

        scores = np.zeros(len(candidates))
        for i, movie_id in enumerate(candidates['movie_id']):
            try:
                score = self.base_engine.cf_model.predict(user_id, movie_id)
                scores[i] = (score + 1) / 2  # Normalize from [-1, 1] to [0, 1]
            except Exception:
                scores[i] = 0.5

        return scores

    def _get_graph_scores(self, candidates: pd.DataFrame, session_positive_movies: List[int]) -> np.ndarray:
        """Get graph co-occurrence scores (0-1 normalized)."""
        if not session_positive_movies or not hasattr(self.base_engine, 'graph'):
            return np.zeros(len(candidates))

        scores = np.zeros(len(candidates))
        for i, movie_id in enumerate(candidates['movie_id']):
            max_similarity = 0
            for pos_movie in session_positive_movies[-5:]:  # Last 5 positive
                try:
                    neighbors = dict(self.base_engine.graph.get_neighbors(pos_movie, top_k=100))
                    if movie_id in neighbors:
                        max_similarity = max(max_similarity, neighbors[movie_id])
                except Exception:
                    pass
            scores[i] = min(1.0, max_similarity)

        return scores

    def _get_session_scores(
        self,
        candidates: pd.DataFrame,
        selected_genres: List[str],
        session_positive_movies: List[int]
    ) -> np.ndarray:
        """Get session-based similarity scores."""
        if not session_positive_movies:
            return np.zeros(len(candidates))

        # Get session movies
        session_movies = self.movies[self.movies['movie_id'].isin(session_positive_movies[-5:])]

        scores = np.zeros(len(candidates))
        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Genre overlap
            movie_genres = set(row['genres']) if isinstance(row['genres'], (list, np.ndarray)) else set()
            selected_set = set(selected_genres)
            genre_overlap = len(movie_genres & selected_set) / len(selected_set) if selected_set else 0

            # Year similarity (within 5 years of session movies)
            if len(session_movies) > 0:
                avg_session_year = session_movies['year'].mean()
                year_diff = abs(row['year'] - avg_session_year)
                year_sim = max(0, 1 - year_diff / 20)  # Decay over 20 years
            else:
                year_sim = 0

            scores[idx] = 0.6 * genre_overlap + 0.4 * year_sim

        return scores

    def _get_quality_scores(self, candidates: pd.DataFrame, evening_type: str) -> np.ndarray:
        """
        Calculate Option C quality scores.

        Uses:
        - Base quality (exponential boost)
        - Confidence (vote count)
        - Evening type modifier
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Get combined rating (average of IMDb + TMDB)
            imdb_rating = row.get('avg_rating', 6.0)
            tmdb_rating = row.get('tmdb_rating', imdb_rating)

            if pd.isna(tmdb_rating):
                combined_rating = imdb_rating
            else:
                combined_rating = (imdb_rating + tmdb_rating) / 2.0

            num_votes = row['num_votes']

            # Calculate Option C score
            scores[idx] = self._calculate_option_c_score(
                combined_rating,
                num_votes,
                evening_type
            )

        return scores

    def _calculate_option_c_score(
        self,
        combined_rating: float,
        num_votes: int,
        evening_type: str
    ) -> float:
        """
        Linear quality scoring with 6.0 floor and smooth confidence curve.

        Candidates already filtered at 6.0, so this spreads them 0.0-1.0:
        - 6.0 rating → 0.0 quality score (minimum acceptable)
        - 7.0 rating → 0.25 quality score
        - 8.0 rating → 0.50 quality score
        - 9.0 rating → 0.75 quality score
        - 10.0 rating → 1.0 quality score

        Returns score between 0.0 and 1.0
        """
        # Step 1: Linear quality (6.0 → 0.0, 10.0 → 1.0)
        base_quality = (combined_rating - 6.0) / 4.0
        base_quality = max(0.0, min(1.0, base_quality))

        # Step 2: Smooth confidence multiplier (log-scale)
        confidence = self._calculate_vote_confidence(num_votes)

        # Step 3: Combine (no evening modifier - quality is quality)
        quality_score = base_quality * confidence

        return quality_score

    def _get_era_scores(self, candidates: pd.DataFrame, era: str) -> np.ndarray:
        """
        Calculate era favorability scores using TimePeriodFilter.

        Movies in preferred era get 1.0, outside get decay based on distance.
        This method is kept for backward compatibility but now uses TimePeriodFilter.
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            scores[idx] = TimePeriodFilter.calculate_era_score(row['year'], era)

        return scores

    def _get_keyword_scores(self, candidates: pd.DataFrame, selected_keywords: List[str]) -> np.ndarray:
        """
        Calculate keyword matching scores.

        Checks both TMDB keywords and descriptions.
        """
        # Filter out metadata keywords that shouldn't affect recommendations
        metadata_keywords = {
            'woman director', 'female director', 'male director',
            'directorial debut', 'independent film', 'film debut'
        }

        filtered_keywords = [kw for kw in selected_keywords if kw.lower() not in metadata_keywords]

        if not filtered_keywords:
            return np.zeros(len(candidates))

        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_dict = row.to_dict()
            scores[idx] = calculate_keyword_match_score(movie_dict, filtered_keywords)

        return scores

    def _get_keyword_scores_enhanced(
        self,
        candidates: pd.DataFrame,
        selected_keywords: Dict[str, Any],
        pacing_pref: Optional[str]
    ) -> np.ndarray:
        """
        Calculate enhanced keyword scores with age, general, adaptation, and pacing.

        Breakdown:
        - Age (35% of keyword score)
        - General keywords (40%)
        - Pacing (15%)
        - Adaptation (10%)
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            components = []

            # Age component (35%)
            if selected_keywords.get('age'):
                age_score = calculate_age_score(row['year'], selected_keywords['age'])
                components.append(age_score * 0.35)

            # General keywords component (40%)
            if selected_keywords.get('general'):
                general_score = self._match_general_keywords(row, selected_keywords['general'])
                components.append(general_score * 0.40)

            # Pacing component (15%)
            if selected_keywords.get('pacing'):
                # Extract pacing keyword value
                pacing_kw = selected_keywords['pacing']
                if isinstance(pacing_kw, list) and len(pacing_kw) > 0:
                    pacing_kw = pacing_kw[0]
                pacing_score = calculate_pacing_match(row, pacing_kw)
                components.append(pacing_score * 0.15)

            # Adaptation component (10%)
            if selected_keywords.get('adaptation'):
                adaptation_score = 1.0 if row.get('is_book_adaptation', False) else 0.0
                components.append(adaptation_score * 0.10)

            # Average components (only if any keywords selected)
            if components:
                scores[idx] = sum(components)

        return scores

    def _match_general_keywords(self, movie_row: pd.Series, user_keywords: List[str]) -> float:
        """
        Match movie against user's general keyword selections.
        Multi-level matching: exact, partial, description.
        """
        if not user_keywords:
            return 0.0

        movie_keywords = movie_row.get('keywords', [])
        description = movie_row.get('description', '')

        # Check if movie_keywords is empty (handle both lists and numpy arrays)
        has_keywords = False
        if isinstance(movie_keywords, (list, np.ndarray)):
            has_keywords = len(movie_keywords) > 0

        if not has_keywords and not description:
            return 0.0

        match_scores = []

        for user_kw in user_keywords:
            # Level 1: Exact match in TMDb keywords
            if has_keywords and isinstance(movie_keywords, (list, np.ndarray)):
                exact_match = any(
                    user_kw.lower() == str(movie_kw).lower()
                    for movie_kw in movie_keywords
                )

                if exact_match:
                    match_scores.append(1.0)
                    continue

                # Level 2: Partial match in TMDb keywords
                partial_match = any(
                    user_kw.lower() in str(movie_kw).lower() or str(movie_kw).lower() in user_kw.lower()
                    for movie_kw in movie_keywords
                )

                if partial_match:
                    match_scores.append(0.7)
                    continue

            # Level 3: Description match
            if description:
                user_kw_formatted = user_kw.replace('_', ' ').lower()
                if user_kw_formatted in description.lower():
                    match_scores.append(0.4)
                    continue

            # No match
            match_scores.append(0.0)

        # Average match score
        return sum(match_scores) / len(user_keywords) if match_scores else 0.0

    def _get_quality_scores_simple(self, candidates: pd.DataFrame) -> np.ndarray:
        """
        Simplified quality scoring without evening modifiers.
        Uses Option C formula: base_quality * confidence
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Base quality (rating normalized to 0-1)
            combined_rating = row['avg_rating']
            base_quality = (combined_rating - 6.0) / 4.0
            base_quality = max(0.0, min(1.0, base_quality))

            # Confidence based on vote count
            num_votes = row['num_votes']
            confidence = self._calculate_vote_confidence(num_votes)

            scores[idx] = base_quality * confidence

        return scores

    def _get_session_scores_enhanced(
        self,
        candidates: pd.DataFrame,
        genres: List[str],
        session_positive_movies: List[int],
        session_history: List[Dict],
        pacing_pref: Optional[str]
    ) -> np.ndarray:
        """
        Enhanced session scoring with pacing awareness.
        """
        if not session_positive_movies:
            return np.ones(len(candidates)) * 0.5

        # Get liked movies data
        liked_movies = [
            item for item in session_history
            if item.get('action') in ['right', 'up']
        ]

        if not liked_movies:
            return np.ones(len(candidates)) * 0.5

        # Extract patterns from liked movies
        liked_genres = set()
        liked_years = []
        liked_pacing = []

        for item in liked_movies:
            movie_id = item['movie_id']
            movie_data = self.movies[self.movies['movie_id'] == movie_id]

            if len(movie_data) > 0:
                movie = movie_data.iloc[0]
                if isinstance(movie.get('genres'), (list, np.ndarray)):
                    liked_genres.update(movie['genres'])
                liked_years.append(movie['year'])

                if 'pacing_score' in item:
                    liked_pacing.append(item['pacing_score'])

        avg_year = np.mean(liked_years) if liked_years else 2015

        # Score candidates
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            score = 0.0

            # Genre overlap (40%)
            if liked_genres:
                movie_genres = set(row['genres']) if isinstance(row.get('genres'), (list, np.ndarray)) else set()
                overlap = len(movie_genres & liked_genres) / len(liked_genres)
                score += overlap * 0.4

            # Year similarity (30%)
            year_diff = abs(row['year'] - avg_year)
            year_score = max(0, 1.0 - (year_diff / 20))
            score += year_score * 0.3

            # Pacing similarity (30%)
            if pacing_pref:
                # Use inferred pacing preference
                movie_pacing = row.get('pacing_score', 0.5)

                if pacing_pref == 'fast' and movie_pacing > 0.7:
                    score += 0.3
                elif pacing_pref == 'slow' and movie_pacing < 0.4:
                    score += 0.3
                else:
                    score += 0.1
            else:
                score += 0.15

            scores[idx] = score

        return scores

    def _calculate_popularity_boosts(
        self,
        candidates: pd.DataFrame,
        popularity_mode: str
    ) -> np.ndarray:
        """
        Calculate popularity boost multipliers.
        """
        boosts = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            boosts[idx] = calculate_popularity_boost(row['num_votes'], popularity_mode)

        return boosts

    def suggest_keywords(self, genres: List[str], num_keywords: int = 8) -> List[str]:
        """
        Suggest contextual keywords based on selected genres.

        Args:
            genres: List of 1-2 selected genres
            num_keywords: Number of keywords to suggest

        Returns:
            List of keyword strings
        """
        if not self.keyword_analyzer:
            return []

        return self.keyword_analyzer.suggest_keywords(genres, num_keywords)

    def _score_candidates_new(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        genres: List[str],
        era: str,
        source_material: str,
        themes: List[str],
        session_positive_movies: List[int],
        session_history: List[Dict]
    ) -> np.ndarray:
        """
        Score candidates with new filtering system.

        Components:
        - Genre matching (30%)
        - Era matching (20%)
        - Source material matching (15%)
        - Theme keywords matching (25%)
        - Quality score (10%)
        """
        num_candidates = len(candidates)
        scores = np.zeros(num_candidates)

        for idx, (_, row) in enumerate(candidates.iterrows()):
            score_components = []

            # 1. Genre matching (40%, increased from 30%)
            movie_genres = set(row['genres']) if isinstance(row['genres'], (list, np.ndarray)) else set()
            genre_set = set(genres)
            genre_overlap = len(movie_genres & genre_set) / len(genre_set) if genre_set else 0
            score_components.append(genre_overlap * 0.40)

            # 2. Era matching (20%, unchanged)
            era_score = TimePeriodFilter.calculate_era_score(row['year'], era)
            score_components.append(era_score * 0.20)

            # 3. Source material REMOVED (was 15%, redistributed to genre +10%, keywords +5%)

            # 4. Theme keywords matching (30%, increased from 25%)
            movie_keywords = row.get('keywords', [])
            if themes:
                theme_matches = 0
                if isinstance(movie_keywords, (list, np.ndarray)):
                    movie_kw_lower = {str(kw).lower() for kw in movie_keywords}
                    for theme in themes:
                        if theme.lower() in movie_kw_lower:
                            theme_matches += 1

                theme_score = theme_matches / len(themes) if themes else 0
                score_components.append(theme_score * 0.30)
            else:
                score_components.append(0.30)  # Full score if no themes specified

            # 5. Quality score (10%, unchanged for now)
            quality = self._calculate_simple_quality(row)
            score_components.append(quality * 0.10)

            scores[idx] = sum(score_components)

        return scores

    def _calculate_simple_quality(self, movie_row: pd.Series) -> float:
        """Simple quality calculation based on rating and votes"""
        rating = movie_row.get('avg_rating', 6.0)
        num_votes = movie_row.get('num_votes', 0)

        # Normalize rating (6.0-10.0 → 0.0-1.0)
        rating_score = (rating - 6.0) / 4.0
        rating_score = max(0.0, min(1.0, rating_score))

        # Vote confidence
        confidence = self._calculate_vote_confidence(num_votes)

        return rating_score * confidence


def load_smart_system(
    models_dir: Path,
    movies_path: Path,
    keyword_db_path: Optional[Path] = None
) -> SmartRecommendationEngine:
    """
    Load complete smart recommendation system.

    Args:
        models_dir: Directory with CF and graph models
        movies_path: Path to movies.parquet
        keyword_db_path: Optional path to keyword_database.pkl

    Returns:
        SmartRecommendationEngine instance
    """
    from recommendation.recommendation_engine import load_system as load_base

    # Load base engine (CF + graph)
    base_engine = load_base(models_dir, movies_path)

    # Load keyword analyzer if available
    keyword_analyzer = None
    if keyword_db_path and keyword_db_path.exists():
        keyword_analyzer = KeywordAnalyzer.load(keyword_db_path)

    # Create smart engine
    smart_engine = SmartRecommendationEngine(
        base_engine=base_engine,
        keyword_analyzer=keyword_analyzer
    )

    return smart_engine


if __name__ == '__main__':
    """Quick test of smart engine."""
    models_dir = Path('output/models')
    movies_path = Path('output/processed/movies.parquet')
    keyword_db_path = Path('output/models/keyword_database.pkl')

    print("Loading smart recommendation system...")
    engine = load_smart_system(models_dir, movies_path, keyword_db_path)

    print("\nTesting recommendations:")

    # Test with keywords
    selected_keywords = {
        'age': 'recent',
        'general': ['espionage', 'heist'],
        'adaptation': False,
        'pacing': ['fast_paced']
    }

    recommendations = engine.recommend(
        user_id=0,
        genres=['action', 'thriller'],
        selected_keywords=selected_keywords,
        session_history=[],
        top_k=10
    )

    print(f"\nTop 10 recommendations (with keywords):")
    for rank, movie_id in enumerate(recommendations, 1):
        movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0]
        print(f"{rank}. {movie['title']} ({movie['year']}) - {movie['avg_rating']:.1f}/10")
