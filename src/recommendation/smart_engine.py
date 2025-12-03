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
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommendation.recommendation_engine import RecommendationEngine
from recommendation.keyword_analyzer import KeywordAnalyzer, calculate_keyword_match_score


class SmartRecommendationEngine:
    """
    Smart recommendation engine with Option C quality scoring and keyword matching.
    """

    # Era configurations
    ERA_CONFIGS = {
        'fresh': {'min_year': 2020, 'max_year': 2025},
        'modern': {'min_year': 2015, 'max_year': 2025},
        'timeless': {'min_year': 2000, 'max_year': 2025},
        'old_school': {'min_year': 1900, 'max_year': 2025}
    }

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

        # Scoring weights
        self.w_cf = 0.30            # Collaborative filtering
        self.w_graph = 0.20         # Co-occurrence graph
        self.w_session = 0.15       # Session similarity
        self.w_quality = 0.15       # Option C quality score
        self.w_era = 0.10           # Era favorability
        self.w_keywords = 0.10      # Keyword matching

    def recommend(
        self,
        user_id: int,
        evening_type: str,
        genres: List[str],
        era: str,
        keywords: Optional[List[str]] = None,
        session_history: Optional[List[Dict]] = None,
        top_k: int = 20
    ) -> List[int]:
        """
        Generate recommendations with smart scoring.

        Args:
            user_id: User ID
            evening_type: 'date_night', 'family_night', 'friends_night', 'chill_evening'
            genres: List of 1-2 selected genres
            era: 'fresh', 'modern', 'timeless', 'old_school'
            keywords: Optional list of 0-2 selected keywords
            session_history: List of {movie_id, action} dicts
            top_k: Number of recommendations to return

        Returns:
            List of movie_ids
        """
        if session_history is None:
            session_history = []
        if keywords is None:
            keywords = []

        # Extract session context
        session_positive = [
            item['movie_id'] for item in session_history
            if item.get('action') in ['right', 'up']
        ]
        already_shown = [item['movie_id'] for item in session_history]

        # STAGE 1: Generate candidates with genre filtering only
        candidates = self._generate_candidates(
            user_id=user_id,
            genres=genres,
            session_positive_movies=session_positive,
            already_shown=already_shown
        )

        if len(candidates) < top_k:
            # Not enough candidates, relax filters
            print(f"Warning: Only {len(candidates)} candidates found, relaxing filters...")
            candidates = self.movies[~self.movies['movie_id'].isin(already_shown)]

        # STAGE 2: Score all candidates
        scores = self._score_candidates(
            candidates=candidates,
            user_id=user_id,
            evening_type=evening_type,
            genres=genres,
            era=era,
            keywords=keywords,
            session_positive_movies=session_positive
        )

        # STAGE 3: Sort and return top K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return candidates.iloc[top_indices]['movie_id'].tolist()

    def _generate_candidates(
        self,
        user_id: int,
        genres: List[str],
        session_positive_movies: List[int],
        already_shown: List[int]
    ) -> pd.DataFrame:
        """
        Generate candidate movies using HARD GENRE FILTER ONLY.
        No quality or era filtering (those are scoring components).

        Returns ~500-1000 candidates.
        """
        candidates = set()

        # Source 1: Genre filtering (hard constraint)
        genre_mask = self.movies['genres'].apply(
            lambda g: any(genre in g for genre in genres) if isinstance(g, (list, np.ndarray)) else False
        )
        genre_movies = self.movies[genre_mask]
        candidates.update(genre_movies['movie_id'].tolist())

        # Source 2: CF top predictions
        if hasattr(self.base_engine, 'cf_model') and self.base_engine.cf_model:
            try:
                cf_predictions = self.base_engine.cf_model.recommend(
                    user_id,
                    top_k=300
                )
                # Filter by genre
                cf_filtered = [mid for mid in cf_predictions if mid in genre_movies['movie_id'].values]
                candidates.update(cf_filtered[:200])
            except:
                pass

        # Source 3: Graph neighbors of session movies
        if session_positive_movies and hasattr(self.base_engine, 'graph'):
            for movie_id in session_positive_movies[-3:]:  # Last 3 positive movies
                try:
                    neighbors = self.base_engine.graph.get_neighbors(movie_id, top_k=50)
                    neighbor_ids = [nid for nid, weight in neighbors]
                    # Filter by genre
                    neighbor_filtered = [mid for mid in neighbor_ids if mid in genre_movies['movie_id'].values]
                    candidates.update(neighbor_filtered[:30])
                except:
                    pass

        # Source 4: Popular movies in genre (cold start)
        popular = genre_movies.nlargest(100, 'num_votes')
        candidates.update(popular['movie_id'].tolist())

        # Remove already shown
        candidates -= set(already_shown)

        # Convert to DataFrame
        candidate_df = self.movies[self.movies['movie_id'].isin(candidates)].copy()

        return candidate_df

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        evening_type: str,
        genres: List[str],
        era: str,
        keywords: List[str],
        session_positive_movies: List[int]
    ) -> np.ndarray:
        """
        Score all candidates using weighted components.

        Returns:
            Array of scores (one per candidate)
        """
        n_candidates = len(candidates)
        scores = np.zeros(n_candidates)

        # Component 1: Collaborative Filtering (30%)
        cf_scores = self._get_cf_scores(candidates, user_id)
        scores += self.w_cf * cf_scores

        # Component 2: Graph similarity (20%)
        graph_scores = self._get_graph_scores(candidates, session_positive_movies)
        scores += self.w_graph * graph_scores

        # Component 3: Session similarity (15%)
        session_scores = self._get_session_scores(candidates, genres, session_positive_movies)
        scores += self.w_session * session_scores

        # Component 4: Quality score - Option C (15%)
        quality_scores = self._get_quality_scores(candidates, evening_type)
        scores += self.w_quality * quality_scores

        # Component 5: Era favorability (10%)
        era_scores = self._get_era_scores(candidates, era)
        scores += self.w_era * era_scores

        # Component 6: Keyword matching (10%)
        if keywords:
            keyword_scores = self._get_keyword_scores(candidates, keywords)
            scores += self.w_keywords * keyword_scores

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
            except:
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
                except:
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
        normal_law_try: Normal distribution quality scoring (centered at 7.5-8.0).

        Uses a normal (Gaussian) distribution to:
        - Prioritize excellent movies (8.0+) and classics
        - Still show good movies (7.0-7.9)
        - Naturally exclude bad movies (<6.0) with very low probability
        - Maintain smooth mathematical curve (no hard cutoffs)

        Returns score between 0.0 and 1.0
        """
        # Step 1: Normal distribution around mean=8.0, std=1.5
        # This creates a bell curve where:
        # - 9.0+ rating → ~1.0 score (masterpieces)
        # - 8.0-8.5 rating → ~0.95-1.0 score (classics/excellent)
        # - 7.0-7.5 rating → ~0.75-0.90 score (very good movies)
        # - 6.5 rating → ~0.55 score (decent movies)
        # - 6.0 rating → ~0.35 score (mediocre)
        # - 5.5 rating → ~0.20 score (rarely shown)
        # - <5.0 rating → ~0.05 score (almost never shown)

        mean = 8.0
        std = 1.5

        # Calculate Gaussian probability
        import math
        z_score = (combined_rating - mean) / std
        gaussian = math.exp(-0.5 * z_score**2)

        # Normalize and boost high ratings
        if combined_rating >= 7.5:
            # Strong boost for excellent movies (7.5+)
            base_quality = min(1.0, gaussian * (1 + (combined_rating - 7.5) * 0.15))
        else:
            # Below 7.5: use pure gaussian (natural decay)
            base_quality = gaussian

        # Step 2: Confidence multiplier
        if num_votes >= 100000:
            confidence = 1.0
        elif num_votes >= 10000:
            confidence = 0.9
        elif num_votes >= 1000:
            confidence = 0.7
        else:
            confidence = 0.5

        # Step 3: Evening type modifier
        modifier = self.EVENING_MODIFIERS.get(evening_type, 1.0)

        # Step 4: Combine
        quality_score = base_quality * confidence * modifier
        quality_score = min(1.0, quality_score)

        return quality_score

    def _get_era_scores(self, candidates: pd.DataFrame, era: str) -> np.ndarray:
        """
        Calculate era favorability scores.

        Movies in preferred era get 1.0, outside get decay (2% per year, min 0.2).
        """
        config = self.ERA_CONFIGS[era]
        min_year = config['min_year']
        max_year = config['max_year']

        scores = np.zeros(len(candidates))

        for idx, year in enumerate(candidates['year'].values):
            if min_year <= year <= max_year:
                scores[idx] = 1.0
            else:
                # Calculate years outside range
                if year < min_year:
                    years_outside = min_year - year
                else:
                    years_outside = year - max_year

                # Decay: 2% per year, minimum 0.2
                decay_rate = 0.02
                scores[idx] = max(0.2, 1.0 - (years_outside * decay_rate))

        return scores

    def _get_keyword_scores(self, candidates: pd.DataFrame, selected_keywords: List[str]) -> np.ndarray:
        """
        Calculate keyword matching scores.

        Checks both TMDB keywords and descriptions.
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_dict = row.to_dict()
            scores[idx] = calculate_keyword_match_score(movie_dict, selected_keywords)

        return scores

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
    recommendations = engine.recommend(
        user_id=0,
        evening_type='date_night',
        genres=['action', 'thriller'],
        era='modern',
        keywords=['espionage'],
        session_history=[],
        top_k=10
    )

    print(f"\nTop 10 recommendations:")
    for rank, movie_id in enumerate(recommendations, 1):
        movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0]
        print(f"{rank}. {movie['title']} ({movie['year']}) - {movie['avg_rating']:.1f}/10")
