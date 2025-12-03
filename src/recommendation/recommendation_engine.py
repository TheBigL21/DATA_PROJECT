"""
RECOMMENDATION ENGINE

Purpose: Generate ranked movie recommendations for users in real-time sessions.

This is the core system that powers the swipe interface.

Input:
- User ID
- Session context:
  * Mood genre (from questionnaire)
  * Mood decade (from questionnaire or inferred)
  * Session history (movies already shown, with actions)

Output:
- Ranked list of movie IDs to show next

Algorithm (Multi-Stage Ranking):

STAGE 1: CANDIDATE GENERATION
  Generate pool of ~500 candidate movies using multiple strategies:

  A. Mood Filtering (Hard Constraint)
     - Filter to movies matching mood genre
     - Filter to movies within mood decade range (±15 years)

  B. Collaborative Filtering Candidates
     - Use CF model to get top-300 movies for this user globally

  C. Session-Based Neighbors
     - For movies user swiped 'right' on in current session:
       - Get graph neighbors (co-occurrence)
       - Add top-100 neighbors to candidates

  D. Popularity Baseline
     - Add top-100 highly-rated movies matching mood (fallback)

STAGE 2: SCORING & RANKING
  For each candidate movie, compute composite score:

  Score = w1 * CF_score
        + w2 * Graph_score
        + w3 * Session_similarity
        - w4 * Diversity_penalty
        + w5 * Rating_boost

  Where:
  - CF_score: collaborative filtering model prediction
  - Graph_score: max edge weight to any session-positive movie
  - Session_similarity: similarity to recent 'right' movies (genre/year overlap)
  - Diversity_penalty: penalize movies too similar to recently shown movies
  - Rating_boost: bonus for highly-rated movies

  Weights (tunable):
  w1=0.4, w2=0.3, w3=0.2, w4=0.1, w5=0.1

STAGE 3: RE-RANKING FOR EXPLORATION
  Apply final adjustments:
  - Boost movies not yet shown in this session
  - Apply diversity: if last 3 movies all same genre, boost different genre
  - Position-based adjustment: early in session, more exploration; later, more exploitation

Returns: Top-K movies ranked by final score
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.collaborative_filtering import ImplicitALS
from models.cooccurrence_graph import MovieCooccurrenceGraph


class RecommendationEngine:
    """
    Main recommendation engine combining CF and graph-based methods.
    """

    def __init__(
        self,
        cf_model: ImplicitALS,
        graph: MovieCooccurrenceGraph,
        movies: pd.DataFrame
    ):
        """
        Args:
            cf_model: Trained collaborative filtering model
            graph: Trained co-occurrence graph
            movies: Movies table with metadata (for filtering)
        """
        self.cf_model = cf_model
        self.graph = graph
        self.movies = movies

        # Create lookup dict for fast movie access
        self.movie_dict = {row['movie_id']: row for _, row in movies.iterrows()}

        # Scoring weights
        self.w_cf = 0.35
        self.w_graph = 0.25
        self.w_session = 0.15
        self.w_diversity = 0.1
        self.w_rating = 0.05
        self.w_popularity = 0.10  # New: popularity weighting

        # Popularity bias parameter (alpha)
        self.popularity_alpha = 0.4

    def generate_candidates(
        self,
        user_id: int,
        mood_genre: str,
        mood_decade: str,
        session_positive_movies: List[int],
        already_shown: set,
        candidate_pool_size: int = 500
    ) -> List[int]:
        """
        STAGE 1: Generate candidate movie pool.

        Args:
            user_id: User ID
            mood_genre: Genre from mood questionnaire
            mood_decade: Decade preference (e.g., '1990s')
            session_positive_movies: Movie IDs user swiped 'right' or 'up' on in this session
            already_shown: Set of movie IDs already shown in this session
            candidate_pool_size: Target size of candidate pool

        Returns:
            List of candidate movie IDs
        """
        candidates = set()

        # Parse decade to year range
        decade_year = int(mood_decade[:4])
        year_min = decade_year - 15
        year_max = decade_year + 15

        # Filter movies by mood
        mood_filtered_movies = self.movies[
            (self.movies['genres'].apply(lambda g: mood_genre in g)) &
            (self.movies['year'] >= year_min) &
            (self.movies['year'] <= year_max)
        ]['movie_id'].values

        # A. Mood-filtered pool (base)
        candidates.update(mood_filtered_movies[:200])

        # B. Collaborative filtering top movies for user
        try:
            cf_candidates = self.cf_model.recommend_for_user(
                user_id,
                top_k=300,
                exclude_movie_ids=already_shown
            )
            # Filter to mood
            cf_candidates_filtered = [m for m in cf_candidates if m in mood_filtered_movies]
            candidates.update(cf_candidates_filtered[:150])
        except:
            pass  # If user not in training set, skip CF

        # C. Graph neighbors of session-positive movies
        if len(session_positive_movies) > 0:
            graph_neighbors = self.graph.get_neighbors_batch(
                session_positive_movies,
                top_k=150
            )
            neighbor_ids = [mid for mid, weight in graph_neighbors]
            # Filter to mood
            neighbor_ids_filtered = [m for m in neighbor_ids if m in mood_filtered_movies]
            candidates.update(neighbor_ids_filtered[:100])

        # D. Popularity baseline (highly-rated movies in mood)
        mood_movies_df = self.movies[self.movies['movie_id'].isin(mood_filtered_movies)]
        top_rated = mood_movies_df.nlargest(100, 'avg_rating')['movie_id'].values
        candidates.update(top_rated[:50])

        # Remove already shown
        candidates -= already_shown

        # If not enough candidates, add POPULARITY-WEIGHTED random mood-filtered movies
        if len(candidates) < candidate_pool_size:
            remaining = set(mood_filtered_movies) - already_shown - candidates
            if len(remaining) > 0:
                # Get popularity weights for remaining movies
                remaining_df = self.movies[self.movies['movie_id'].isin(remaining)]

                if len(remaining_df) > 0:
                    # Calculate popularity weights: (num_votes / max_votes)^alpha
                    max_votes = remaining_df['num_votes'].max()
                    if max_votes > 0:
                        weights = (remaining_df['num_votes'] / max_votes) ** self.popularity_alpha
                        weights = weights / weights.sum()  # Normalize to probabilities

                        # Sample with popularity weighting
                        n_to_sample = min(len(remaining), candidate_pool_size - len(candidates))
                        sampled = np.random.choice(
                            remaining_df['movie_id'].values,
                            size=n_to_sample,
                            replace=False,
                            p=weights
                        )
                        candidates.update(sampled)
                    else:
                        # Fallback: uniform random if no votes data
                        candidates.update(np.random.choice(
                            list(remaining),
                            size=min(len(remaining), candidate_pool_size - len(candidates)),
                            replace=False
                        ))

        return list(candidates)

    def score_candidates(
        self,
        user_id: int,
        candidate_ids: List[int],
        session_positive_movies: List[int],
        recently_shown: List[int]
    ) -> np.ndarray:
        """
        STAGE 2: Score candidate movies using composite scoring function.

        Args:
            user_id: User ID
            candidate_ids: List of candidate movie IDs
            session_positive_movies: Movies user swiped right/up on in this session
            recently_shown: Last N movies shown (for diversity penalty)

        Returns:
            scores: Array of scores for each candidate (same order as candidate_ids)
        """
        scores = np.zeros(len(candidate_ids))

        for idx, movie_id in enumerate(candidate_ids):
            movie = self.movie_dict[movie_id]

            # Component 1: Collaborative Filtering score
            try:
                cf_score = self.cf_model.predict(user_id, movie_id)
            except:
                cf_score = 0.5  # Default for new users

            # Component 2: Graph score (max edge weight to session-positive movies)
            graph_score = 0.0
            if len(session_positive_movies) > 0:
                for pos_movie_id in session_positive_movies:
                    neighbors = self.graph.get_neighbors(pos_movie_id, top_k=200)
                    for neighbor_id, weight in neighbors:
                        if neighbor_id == movie_id:
                            graph_score = max(graph_score, weight)
                            break

            # Component 3: Session similarity (genre overlap with session-positive movies)
            session_sim = 0.0
            if len(session_positive_movies) > 0:
                movie_genres = set(movie['genres'])
                for pos_movie_id in session_positive_movies:
                    pos_movie = self.movie_dict[pos_movie_id]
                    pos_genres = set(pos_movie['genres'])
                    overlap = len(movie_genres & pos_genres)
                    if len(movie_genres) > 0:
                        session_sim += overlap / len(movie_genres)
                session_sim /= len(session_positive_movies)

            # Component 4: Diversity penalty (penalize if too similar to recently shown)
            diversity_penalty = 0.0
            if len(recently_shown) > 0:
                movie_genres = set(movie['genres'])
                for recent_id in recently_shown[-5:]:  # Last 5 shown
                    recent_movie = self.movie_dict[recent_id]
                    recent_genres = set(recent_movie['genres'])
                    overlap = len(movie_genres & recent_genres)
                    if len(movie_genres) > 0:
                        diversity_penalty += overlap / len(movie_genres)
                diversity_penalty /= min(len(recently_shown), 5)

            # Component 5: Rating boost (normalized to 0-1)
            rating_boost = (movie['avg_rating'] - 5.0) / 5.0  # Assuming ratings 0-10

            # Component 6: Popularity score (normalized to 0-1)
            # Get max votes from all candidates for normalization
            max_votes_in_pool = max(self.movie_dict[cid]['num_votes'] for cid in candidate_ids)
            if max_votes_in_pool > 0:
                popularity_score = (movie['num_votes'] / max_votes_in_pool) ** self.popularity_alpha
            else:
                popularity_score = 0.5  # Default if no vote data

            # Composite score
            score = (
                self.w_cf * cf_score +
                self.w_graph * graph_score +
                self.w_session * session_sim -
                self.w_diversity * diversity_penalty +
                self.w_rating * rating_boost +
                self.w_popularity * popularity_score
            )

            scores[idx] = score

        return scores

    def recommend(
        self,
        user_id: int,
        mood_genre: str,
        mood_decade: str,
        session_history: List[Dict],
        top_k: int = 20
    ) -> List[int]:
        """
        Main recommendation method: generate top-K movies for next swipes.

        Args:
            user_id: User ID
            mood_genre: Genre from questionnaire
            mood_decade: Decade preference
            session_history: List of dicts with keys [movie_id, action]
                             e.g., [{'movie_id': 42, 'action': 'left'}, ...]
            top_k: Number of recommendations to return

        Returns:
            List of movie IDs ranked by score
        """
        # Extract session state
        already_shown = set([item['movie_id'] for item in session_history])
        session_positive_movies = [
            item['movie_id'] for item in session_history
            if item['action'] in ['right', 'up']
        ]
        recently_shown = [item['movie_id'] for item in session_history[-10:]]

        # Stage 1: Generate candidates
        candidates = self.generate_candidates(
            user_id,
            mood_genre,
            mood_decade,
            session_positive_movies,
            already_shown,
            candidate_pool_size=500
        )

        if len(candidates) == 0:
            # Fallback: return random popular movies
            return self.movies.nlargest(top_k, 'avg_rating')['movie_id'].tolist()

        # Stage 2: Score candidates
        scores = self.score_candidates(
            user_id,
            candidates,
            session_positive_movies,
            recently_shown
        )

        # Stage 3: Re-rank with exploration adjustments
        # Early in session: add randomness
        # Late in session: more deterministic
        session_position = len(session_history)
        if session_position < 5:
            # Early: add noise for exploration
            noise = np.random.normal(0, 0.2, size=len(scores))
            scores += noise

        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]

        return ranked_candidates[:top_k]


def load_system(models_dir: Path, movies_path: Path) -> RecommendationEngine:
    """
    Load all components and initialize recommendation engine.

    Args:
        models_dir: Directory containing trained CF model and graph
        movies_path: Path to movies.parquet

    Returns:
        RecommendationEngine instance
    """
    print("Loading collaborative filtering model...")
    cf_model = ImplicitALS.load(models_dir)

    print("Loading co-occurrence graph...")
    graph = MovieCooccurrenceGraph.load(models_dir)

    print("Loading movies...")
    movies = pd.read_parquet(movies_path)

    print("Initializing recommendation engine...")
    engine = RecommendationEngine(cf_model, graph, movies)

    print("System ready.")
    return engine


def main():
    """
    Demo: Load system and generate recommendations for a test user.
    """
    # Paths (adjust as needed)
    models_dir = Path("../../output/models")
    movies_path = Path("../../output/processed/movies.parquet")

    # Load system
    engine = load_system(models_dir, movies_path)

    # Test: Generate recommendations for user 0
    print("\n=== TEST RECOMMENDATION ===")
    user_id = 0
    mood_genre = "action"
    mood_decade = "2000s"
    session_history = [
        {'movie_id': 100, 'action': 'left'},
        {'movie_id': 250, 'action': 'right'},
        {'movie_id': 380, 'action': 'left'},
    ]

    recommendations = engine.recommend(
        user_id,
        mood_genre,
        mood_decade,
        session_history,
        top_k=10
    )

    print(f"\nUser {user_id}, Mood: {mood_genre} {mood_decade}")
    print(f"Session history: {len(session_history)} interactions")
    print(f"\nTop 10 recommendations:")
    for rank, movie_id in enumerate(recommendations, 1):
        movie = engine.movie_dict[movie_id]
        print(f"{rank}. {movie['title']} ({movie['year']}) - {movie['genres'][:3]} - Rating: {movie['avg_rating']:.1f}")


if __name__ == '__main__':
    main()
