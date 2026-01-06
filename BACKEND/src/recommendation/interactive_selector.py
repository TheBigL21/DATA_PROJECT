"""
INTERACTIVE MOVIE SELECTOR WITH ADAPTIVE LEARNING

Presents one movie at a time from evolving candidate pool.
Learns from user feedback to adapt both candidates and ranking.

User Actions:
- 0 or Enter → "yes" (like but not now, want similar) - satisfaction: 0.3
- 1 → "no" (reject, shift away) - satisfaction: 0.0
- 2 → "final" (accept and watch) - satisfaction: 1.0

Learning Components:
1. Session-level: Feature weights updated per action
2. Sequence-level: Which movies work well together
3. Global-level: Movie quality by context across all users
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import pickle
import json
from collections import defaultdict
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from time_period_filter import TimePeriodFilter


# ============================================================================
# CONSTANTS
# ============================================================================

# Era year thresholds
ERA_YEAR_CLASSIC = 1980
ERA_YEAR_OLD = 2000
ERA_YEAR_MODERN = 2015

# Quality thresholds (based on IMDb rating)
QUALITY_THRESHOLD_MEDIUM = 6.5
QUALITY_THRESHOLD_HIGH = 7.5

# Pacing thresholds (based on runtime/rating ratio)
PACING_THRESHOLD_SLOW = 0.4
PACING_THRESHOLD_FAST = 0.7

# Similarity thresholds
SIMILARITY_THRESHOLD_EXPAND = 0.6    # Add very similar movies
SIMILARITY_THRESHOLD_REMOVE = 0.3    # Remove very dissimilar movies
DIVERSITY_THRESHOLD_REMOVE = 0.5     # Remove somewhat similar movies

# Scoring weights
WEIGHT_STRICT = 0.80       # Weight for strict (SmartEngine) score
WEIGHT_ADAPTIVE = 0.20     # Weight for adaptive (learning) score
WEIGHT_SESSION = 0.50      # Session learning contribution to adaptive
WEIGHT_SEQUENCE = 0.30     # Sequence compatibility contribution to adaptive
WEIGHT_GLOBAL = 0.20       # Global quality contribution to adaptive
ADAPTIVE_RANGE = 0.6       # Maximum adaptive score adjustment (+/-)

# Session learner
SESSION_LEARNING_RATE = 0.2


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FeedbackEvent:
    """User feedback event for learning"""
    user_id: int
    session_id: str
    timestamp: str
    context: Dict[str, Any]  # Original 4-question answers
    movie_id: int
    action: str  # 'yes', 'no', 'final'
    satisfaction: float  # 0.0, 0.3, 1.0
    position_in_session: int
    previous_movie_id: Optional[int] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['context'] = json.dumps(d['context'])
        return d


# ============================================================================
# FEATURE ENCODING & SIMILARITY
# ============================================================================

class FeatureEncoder:
    """Encode movies as binary feature vectors for similarity computation"""

    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        self.movie_dict = {row['movie_id']: row for _, row in movies_df.iterrows()}

        # Extract vocabulary
        all_genres = set()
        all_keywords = set()

        for _, row in movies_df.iterrows():
            if isinstance(row.get('genres'), (list, np.ndarray)):
                all_genres.update(row['genres'])
            if isinstance(row.get('keywords'), (list, np.ndarray)):
                all_keywords.update([str(k).lower() for k in row['keywords']])

        self.all_genres = sorted(list(all_genres))
        self.top_keywords = sorted(list(all_keywords))[:100]  # Top 100 keywords

        # Feature name mapping for interpretability
        self.feature_names = []
        self.feature_names.extend([f'genre_{g}' for g in self.all_genres])
        self.feature_names.extend([f'keyword_{k}' for k in self.top_keywords])
        self.feature_names.extend(['era_classic', 'era_old', 'era_modern', 'era_recent'])
        self.feature_names.extend(['pacing_slow', 'pacing_medium', 'pacing_fast'])
        self.feature_names.extend(['quality_low', 'quality_medium', 'quality_high'])

    def encode(self, movie_id: int) -> np.ndarray:
        """
        Encode movie as binary feature vector.

        Returns:
            Binary vector with features activated
        """
        movie = self.movie_dict[movie_id]
        features = []

        # Genre multi-hot
        for genre in self.all_genres:
            has_genre = 0
            if isinstance(movie.get('genres'), (list, np.ndarray)):
                has_genre = 1 if genre in movie['genres'] else 0
            features.append(has_genre)

        # Keyword multi-hot
        movie_keywords = []
        if isinstance(movie.get('keywords'), (list, np.ndarray)):
            movie_keywords = [str(k).lower() for k in movie['keywords']]

        for keyword in self.top_keywords:
            features.append(1 if keyword in movie_keywords else 0)

        # Era bins (classic, old, modern, recent)
        year = movie['year']
        features.append(1 if year < ERA_YEAR_CLASSIC else 0)  # Classic
        features.append(1 if ERA_YEAR_CLASSIC <= year < ERA_YEAR_OLD else 0)  # Old
        features.append(1 if ERA_YEAR_OLD <= year < ERA_YEAR_MODERN else 0)  # Modern
        features.append(1 if year >= ERA_YEAR_MODERN else 0)  # Recent

        # Pacing bins
        pacing = movie.get('pacing_score', 0.5)
        features.append(1 if pacing < PACING_THRESHOLD_SLOW else 0)  # Slow
        features.append(1 if PACING_THRESHOLD_SLOW <= pacing <= PACING_THRESHOLD_FAST else 0)  # Medium
        features.append(1 if pacing > PACING_THRESHOLD_FAST else 0)  # Fast

        # Quality bins
        rating = movie.get('avg_rating', 6.0)
        features.append(1 if rating < QUALITY_THRESHOLD_MEDIUM else 0)  # Low
        features.append(1 if QUALITY_THRESHOLD_MEDIUM <= rating < QUALITY_THRESHOLD_HIGH else 0)  # Medium
        features.append(1 if rating >= QUALITY_THRESHOLD_HIGH else 0)  # High

        return np.array(features, dtype=float)

    def compute_similarity(self, movie_a_id: int, movie_b_id: int) -> float:
        """Cosine similarity between two movies"""
        vec_a = self.encode(movie_a_id)
        vec_b = self.encode(movie_b_id)

        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_active_features(self, movie_id: int) -> List[str]:
        """Get list of active feature names for a movie (for debugging)"""
        vec = self.encode(movie_id)
        return [self.feature_names[i] for i, val in enumerate(vec) if val > 0]


# ============================================================================
# LEARNING MODELS
# ============================================================================

class SessionLearner:
    """Learn feature preferences within a session"""

    def __init__(self, feature_names: List[str], learning_rate: float = SESSION_LEARNING_RATE):
        self.feature_names = feature_names
        self.learning_rate = learning_rate

        # Feature weights (positive = good, negative = bad)
        self.weights = np.zeros(len(feature_names))
        self.n_updates = 0

    def update(self, feature_vector: np.ndarray, action: str):
        """
        Update weights based on user action.

        Args:
            feature_vector: Binary features of shown movie
            action: 'yes', 'no', 'final'
        """
        if action == 'yes':
            # User likes these features → increase weights
            self.weights += self.learning_rate * feature_vector
        elif action == 'no':
            # User dislikes these features → decrease weights
            self.weights -= self.learning_rate * feature_vector
        elif action == 'final':
            # Strong positive signal
            self.weights += (self.learning_rate * 2.0) * feature_vector

        self.n_updates += 1

    def score_movie(self, feature_vector: np.ndarray) -> float:
        """Score a movie based on learned weights"""
        if self.n_updates == 0:
            return 0.0  # Neutral when no learning yet

        return np.dot(self.weights, feature_vector)

    def get_top_features(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top positive and negative features"""
        indices = np.argsort(np.abs(self.weights))[-top_k:][::-1]
        return [(self.feature_names[i], self.weights[i]) for i in indices]


class SequenceLearner:
    """Learn which movie pairs work well together"""

    def __init__(self):
        # (movie_a, movie_b) → [count, success_count, total_satisfaction]
        self.transitions: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0, 0, 0.0])

    def update(self, movie_a_id: int, movie_b_id: int, satisfaction: float):
        """Record outcome of showing movie_b after movie_a"""
        key = (movie_a_id, movie_b_id)
        stats = self.transitions[key]

        stats[0] += 1  # count
        if satisfaction >= 0.3:  # 'yes' or 'final'
            stats[1] += 1  # success
        stats[2] += satisfaction  # total satisfaction

    def get_compatibility(self, movie_a_id: int, movie_b_id: int) -> float:
        """
        Get compatibility score (0-1) for showing movie_b after movie_a.

        Returns empirical success rate, or 0.5 if insufficient data.
        """
        key = (movie_a_id, movie_b_id)

        if key not in self.transitions:
            return 0.5  # Neutral

        stats = self.transitions[key]
        count = stats[0]

        if count < 2:  # Need at least 2 observations
            return 0.5

        success_rate = stats[1] / count
        return success_rate


class GlobalLearner:
    """Learn movie quality across all users by context"""

    def __init__(self):
        # context_key → movie_id → [count, success_count, total_satisfaction]
        self.context_movie_stats: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0, 0.0])
        )

        # Global movie stats (context-independent)
        # movie_id → [count, success_count, total_satisfaction]
        self.global_movie_stats: Dict[int, List[float]] = defaultdict(lambda: [0, 0, 0.0])

    def _context_key(self, context: Dict[str, Any]) -> str:
        """Create context key from user's 4 answers"""
        genres = sorted(context.get('genres', []))
        return f"{','.join(genres)}|{context.get('era', 'any')}|{context.get('source_material', 'any')}"

    def update(self, context: Dict[str, Any], movie_id: int, satisfaction: float):
        """Update statistics for movie in context"""
        ctx_key = self._context_key(context)

        # Update context-specific stats
        stats = self.context_movie_stats[ctx_key][movie_id]
        stats[0] += 1  # count
        if satisfaction >= 0.3:
            stats[1] += 1  # success
        stats[2] += satisfaction

        # Update global stats
        global_stats = self.global_movie_stats[movie_id]
        global_stats[0] += 1
        if satisfaction >= 0.3:
            global_stats[1] += 1
        global_stats[2] += satisfaction

    def get_movie_penalty(self, context: Dict[str, Any], movie_id: int) -> float:
        """
        Get penalty/boost for a movie in context.

        Returns:
            Penalty between -0.5 (bad movie) and +0.5 (good movie)
        """
        ctx_key = self._context_key(context)

        # Check context-specific stats first
        if ctx_key in self.context_movie_stats and movie_id in self.context_movie_stats[ctx_key]:
            stats = self.context_movie_stats[ctx_key][movie_id]
            count = stats[0]

            if count >= 5:  # Need 5+ observations
                success_rate = stats[1] / count
                # Map success rate to penalty: 0% → -0.5, 50% → 0.0, 100% → +0.5
                return (success_rate - 0.5)

        # Fallback to global stats
        if movie_id in self.global_movie_stats:
            stats = self.global_movie_stats[movie_id]
            count = stats[0]

            if count >= 10:  # Need 10+ global observations
                success_rate = stats[1] / count
                return (success_rate - 0.5) * 0.5  # Weaker signal

        return 0.0  # Neutral

    def get_worst_movies(self, context: Dict[str, Any], top_k: int = 10) -> List[Tuple[int, float]]:
        """Get movies with worst success rates in this context"""
        ctx_key = self._context_key(context)

        if ctx_key not in self.context_movie_stats:
            return []

        movie_scores = []
        for movie_id, stats in self.context_movie_stats[ctx_key].items():
            count, success_count, _ = stats
            if count >= 5:
                success_rate = success_count / count
                movie_scores.append((movie_id, success_rate))

        # Sort by success rate ascending
        movie_scores.sort(key=lambda x: x[1])
        return movie_scores[:top_k]


# ============================================================================
# ADAPTIVE POOL MANAGER
# ============================================================================

class AdaptivePoolManager:
    """Manage evolving candidate pool based on user feedback"""

    def __init__(
        self,
        encoder: FeatureEncoder,
        movies_df: pd.DataFrame,
        initial_candidates: List[int],
        context: Dict[str, Any],
        pool_size_target: int = 80
    ):
        self.encoder = encoder
        self.movies_df = movies_df
        self.context = context
        self.pool_size_target = pool_size_target

        # Active pool
        self.pool = set(initial_candidates[:pool_size_target])
        self.shown_movies = set()

        # Full filtered catalog (respects context constraints)
        self.available_catalog = self._build_context_catalog()

    def _build_context_catalog(self) -> set:
        """Build catalog of all movies matching original context with HARD FILTERS"""
        genres = self.context.get('genres', [])
        era = self.context.get('era', 'any')
        source_material = self.context.get('source_material', 'any')
        themes = self.context.get('themes', [])

        # Filter movies
        mask = pd.Series([True] * len(self.movies_df))

        # HARD FILTER 1: Genre filter (EXACT match required - at least one selected genre)
        if genres:
            mask &= self.movies_df['genres'].apply(
                lambda g: any(genre in g for genre in genres) if isinstance(g, (list, np.ndarray)) else False
            )

        # HARD FILTER 2: Era filter (STRICT year ranges - no movies outside range)
        if era != 'any':
            year_min, year_max = TimePeriodFilter.get_year_range(era)

            if year_min is not None and year_max is not None:
                mask &= (self.movies_df['year'] >= year_min) & (self.movies_df['year'] <= year_max)

        # Source material filter (soft - only if explicitly chosen)
        if source_material == 'book':
            mask &= self.movies_df['is_book_adaptation'] == True

        # Themes/Keywords filter (soft - only if themes specified)
        if themes:
            def has_theme_keyword(movie_keywords):
                if not isinstance(movie_keywords, (list, np.ndarray)):
                    return False
                movie_kw_lower = [str(k).lower() for k in movie_keywords]
                return any(theme.lower() in movie_kw_lower for theme in themes)

            mask &= self.movies_df['keywords'].apply(has_theme_keyword)

        return set(self.movies_df[mask]['movie_id'].tolist())

    def get_available(self) -> List[int]:
        """Get movies from pool not yet shown"""
        return list(self.pool - self.shown_movies)

    def mark_shown(self, movie_id: int):
        """Mark movie as shown"""
        self.shown_movies.add(movie_id)

    def expand_similar(self, reference_movie_id: int, similarity_threshold: float = SIMILARITY_THRESHOLD_EXPAND, add_count: int = 20):
        """
        Add movies similar to reference (after 'yes' action).
        Remove dissimilar movies from pool.
        """
        # Find similar movies from catalog
        similar_movies = []

        for movie_id in self.available_catalog:
            if movie_id in self.pool or movie_id in self.shown_movies:
                continue

            similarity = self.encoder.compute_similarity(reference_movie_id, movie_id)
            if similarity >= similarity_threshold:
                similar_movies.append((movie_id, similarity))

        # Sort by similarity descending
        similar_movies.sort(key=lambda x: x[1], reverse=True)

        # Add top similar movies
        for movie_id, _ in similar_movies[:add_count]:
            self.pool.add(movie_id)

        # Remove dissimilar movies from pool
        to_remove = []
        for movie_id in self.pool:
            if movie_id in self.shown_movies:
                continue

            similarity = self.encoder.compute_similarity(reference_movie_id, movie_id)
            if similarity < SIMILARITY_THRESHOLD_REMOVE:  # Very dissimilar
                to_remove.append(movie_id)

        for movie_id in to_remove[:10]:  # Remove at most 10
            self.pool.discard(movie_id)

    def expand_diverse(self, avoid_movie_id: int, diversity_threshold: float = DIVERSITY_THRESHOLD_REMOVE, add_count: int = 20):
        """
        Add movies diverse from avoid_movie (after 'no' action).
        Remove similar movies from pool.
        """
        # Remove similar movies from pool
        to_remove = []
        for movie_id in self.pool:
            if movie_id in self.shown_movies:
                continue

            similarity = self.encoder.compute_similarity(avoid_movie_id, movie_id)
            if similarity >= diversity_threshold:
                to_remove.append(movie_id)

        for movie_id in to_remove:
            self.pool.discard(movie_id)

        # Find diverse movies from catalog
        diverse_movies = []

        for movie_id in self.available_catalog:
            if movie_id in self.pool or movie_id in self.shown_movies:
                continue

            similarity = self.encoder.compute_similarity(avoid_movie_id, movie_id)
            if similarity < diversity_threshold:
                diverse_movies.append((movie_id, similarity))

        # Sort by similarity ascending (most diverse first)
        diverse_movies.sort(key=lambda x: x[1])

        # Add diverse movies
        for movie_id, _ in diverse_movies[:add_count]:
            self.pool.add(movie_id)

    def maintain_size(self):
        """Ensure pool stays around target size"""
        current_available = len(self.get_available())

        if current_available < self.pool_size_target // 2:
            # Pool too small, add random movies from catalog
            needed = self.pool_size_target - current_available

            candidates = list(self.available_catalog - self.pool - self.shown_movies)
            if candidates:
                to_add = np.random.choice(
                    candidates,
                    size=min(needed, len(candidates)),
                    replace=False
                )
                self.pool.update(to_add)


# ============================================================================
# INTERACTIVE SELECTOR
# ============================================================================

class InteractiveSelector:
    """Main orchestrator for interactive movie selection"""

    def __init__(
        self,
        encoder: FeatureEncoder,
        movies_df: pd.DataFrame,
        session_learner: SessionLearner,
        sequence_learner: SequenceLearner,
        global_learner: GlobalLearner,
        seed: int = 42
    ):
        self.encoder = encoder
        self.movies_df = movies_df
        self.movie_dict = {row['movie_id']: row for _, row in movies_df.iterrows()}

        self.session_learner = session_learner
        self.sequence_learner = sequence_learner
        self.global_learner = global_learner

        self.rng = np.random.RandomState(seed)

        # TWO-STAGE SCORING WEIGHTS (80% strict, 20% adaptive)
        self.w_strict = WEIGHT_STRICT  # Stage 1 strict score (preserved constraints)
        self.w_adaptive = WEIGHT_ADAPTIVE  # Stage 2 adaptive refinement

        # Adaptive layer component weights (within 20% budget)
        self.w_session = WEIGHT_SESSION  # Session learning weight
        self.w_sequence = WEIGHT_SEQUENCE  # Sequence compatibility weight
        self.w_global = WEIGHT_GLOBAL  # Global quality weight

        # Adaptive score dampening (prevent extreme swings)
        self.adaptive_range = ADAPTIVE_RANGE  # Dampen from ±1.0 to ±0.6

        # Store strict scores for each movie
        self.strict_scores = {}

    def set_strict_scores(self, strict_scores: Dict[int, float]):
        """
        Set strict scores from Stage 1 (SmartEngine).

        Args:
            strict_scores: Dict mapping movie_id → strict score [0.0, 1.0]
        """
        self.strict_scores = strict_scores

    def score_candidates(
        self,
        candidates: List[int],
        context: Dict[str, Any],
        last_movie_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Score all candidate movies using 80/20 split (strict/adaptive).

        Formula:
            final_score = (strict_score × 0.80) + (adaptive_score × 0.20)

        Where adaptive_score combines:
            - Session learning (50% of adaptive budget)
            - Sequence compatibility (30% of adaptive budget)
            - Global quality (20% of adaptive budget)

        Returns:
            Array of scores (one per candidate)
        """
        scores = np.zeros(len(candidates))

        for i, movie_id in enumerate(candidates):
            # STAGE 1: Get strict score (from SmartEngine)
            strict_score = self.strict_scores.get(movie_id, 0.5)  # Default 0.5 if missing

            # STAGE 2: Calculate adaptive refinement
            feature_vec = self.encoder.encode(movie_id)

            # Component 1: Session learning score (normalized to [-1, 1])
            session_raw = self.session_learner.score_movie(feature_vec)
            session_norm = np.tanh(session_raw / 5.0)  # Normalize unbounded score

            # Component 2: Sequence compatibility ([-1, 1])
            sequence_score = 0.0
            if last_movie_id is not None:
                compat = self.sequence_learner.get_compatibility(last_movie_id, movie_id)
                sequence_score = (compat - 0.5) * 2.0  # Map [0,1] to [-1,1]

            # Component 3: Global quality penalty/boost ([-0.5, 0.5])
            global_penalty = self.global_learner.get_movie_penalty(context, movie_id)

            # Combine adaptive components
            adaptive_raw = (
                session_norm * self.w_session +
                sequence_score * self.w_sequence +
                global_penalty * 2.0 * self.w_global  # Scale global to [-1, 1] range
            )

            # Dampen adaptive score to prevent extreme swings
            adaptive_score = np.clip(adaptive_raw, -self.adaptive_range, self.adaptive_range)

            # FINAL SCORE: 80% strict + 20% adaptive
            final_score = (strict_score * self.w_strict) + (adaptive_score * self.w_adaptive)

            scores[i] = final_score

        return scores

    def select_next(
        self,
        pool_manager: AdaptivePoolManager,
        context: Dict[str, Any],
        last_movie_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Select next best movie from pool.

        Returns:
            movie_id or None if pool exhausted
        """
        available = pool_manager.get_available()

        if len(available) == 0:
            return None

        # Score all available movies
        scores = self.score_candidates(available, context, last_movie_id)

        # Select best
        best_idx = np.argmax(scores)
        return available[best_idx]

    def process_feedback(
        self,
        movie_id: int,
        action: str,
        context: Dict[str, Any],
        pool_manager: AdaptivePoolManager,
        last_movie_id: Optional[int] = None
    ):
        """
        Process user feedback and update all learning components.

        Args:
            movie_id: Shown movie
            action: 'yes', 'no', 'final'
            context: User context
            pool_manager: Pool to adapt
            last_movie_id: Previous movie (for sequence learning)
        """
        # Map action to satisfaction
        satisfaction = action_to_satisfaction(action)

        # Get features
        feature_vec = self.encoder.encode(movie_id)

        # Update session learner
        self.session_learner.update(feature_vec, action)

        # Update sequence learner
        if last_movie_id is not None:
            self.sequence_learner.update(last_movie_id, movie_id, satisfaction)

        # Update global learner
        self.global_learner.update(context, movie_id, satisfaction)

        # Adapt pool based on action
        if action == 'yes':
            pool_manager.expand_similar(movie_id, similarity_threshold=0.6, add_count=15)
        elif action == 'no':
            pool_manager.expand_diverse(movie_id, diversity_threshold=0.5, add_count=15)

        # Maintain pool size
        pool_manager.maintain_size()


# ============================================================================
# STORAGE
# ============================================================================

class LearningStorage:
    """Persist learning models"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Event log
        self.events: List[FeedbackEvent] = []

    def log_event(self, event: FeedbackEvent):
        """Log feedback event"""
        self.events.append(event)

    def save_models(
        self,
        sequence_learner: SequenceLearner,
        global_learner: GlobalLearner
    ):
        """Save persistent models (sequence + global)"""
        models = {
            'sequence': {
                'transitions': {f'{k[0]}_{k[1]}': v for k, v in sequence_learner.transitions.items()}
            },
            'global': {
                'context_movie_stats': {
                    ctx_key: dict(movie_stats)
                    for ctx_key, movie_stats in global_learner.context_movie_stats.items()
                },
                'global_movie_stats': dict(global_learner.global_movie_stats)
            }
        }

        save_path = self.storage_dir / 'learning_models.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(models, f)

    def load_models(self) -> Tuple[SequenceLearner, GlobalLearner]:
        """Load persistent models"""
        save_path = self.storage_dir / 'learning_models.pkl'

        if not save_path.exists():
            return SequenceLearner(), GlobalLearner()

        with open(save_path, 'rb') as f:
            models = pickle.load(f)

        # Reconstruct sequence learner
        sequence_learner = SequenceLearner()
        for key_str, stats in models['sequence']['transitions'].items():
            movie_a, movie_b = map(int, key_str.split('_'))
            sequence_learner.transitions[(movie_a, movie_b)] = stats

        # Reconstruct global learner
        global_learner = GlobalLearner()
        global_learner.context_movie_stats = defaultdict(
            lambda: defaultdict(lambda: [0, 0, 0.0]),
            {
                ctx_key: defaultdict(lambda: [0, 0, 0.0], movie_stats)
                for ctx_key, movie_stats in models['global']['context_movie_stats'].items()
            }
        )
        global_learner.global_movie_stats = defaultdict(
            lambda: [0, 0, 0.0],
            models['global']['global_movie_stats']
        )

        return sequence_learner, global_learner


# ============================================================================
# UTILITIES
# ============================================================================

def action_to_satisfaction(action: str) -> float:
    """Map user action to satisfaction score"""
    mapping = {
        'yes': 0.3,
        'no': 0.0,
        'final': 1.0
    }
    return mapping[action]


def display_movie_details(movie: pd.Series):
    """Display full movie details for final choice"""
    print("\n" + "="*80)
    print("YOUR MOVIE CHOICE:")
    print("="*80)
    print(f"\nTitle: {movie['title']}")
    print(f"Year: {movie['year']}")
    print(f"Rating: {movie['avg_rating']:.1f}/10 ({movie['num_votes']:,} votes)")

    if isinstance(movie.get('genres'), (list, np.ndarray)):
        print(f"Genres: {', '.join(movie['genres'])}")

    print(f"Runtime: {movie.get('runtime', 'N/A')} minutes")
    print(f"Director: {movie.get('director', 'N/A')}")

    if isinstance(movie.get('actors'), (list, np.ndarray)) and len(movie['actors']) > 0:
        print(f"Cast: {', '.join(movie['actors'][:5])}")

    if isinstance(movie.get('keywords'), (list, np.ndarray)) and len(movie['keywords']) > 0:
        print(f"Keywords: {', '.join([str(k) for k in movie['keywords'][:10]])}")

    if movie.get('description'):
        print(f"\nDescription:")
        print(f"{movie['description']}")

    if movie.get('poster_url'):
        print(f"\nPoster: {movie['poster_url']}")

    print("\n" + "="*80)
    print("Enjoy your movie!")
    print("="*80 + "\n")


def display_movie_card(movie: pd.Series, position: int):
    """Display movie card for user choice"""
    print("\n" + "-"*60)
    print(f"Movie #{position}")
    print("-"*60)
    print(f"{movie['title']} ({movie['year']})")
    print(f"Rating: {movie['avg_rating']:.1f}/10")

    if isinstance(movie.get('genres'), (list, np.ndarray)):
        print(f"Genres: {', '.join(movie['genres'][:3])}")

    if isinstance(movie.get('keywords'), (list, np.ndarray)) and len(movie['keywords']) > 0:
        keywords_display = ', '.join([str(k) for k in movie['keywords'][:5]])
        print(f"Keywords: {keywords_display}")

    print("-"*60)
    print("\nYour choice:")
    print("  0 or Enter → Yes (like it, show similar)")
    print("  1 → No (not interested, show different)")
    print("  2 → FINAL (choose this movie!)")
    print()
