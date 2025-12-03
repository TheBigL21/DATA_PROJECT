"""
COLLABORATIVE FILTERING MODEL

Purpose: Learn latent factors for users and movies from implicit feedback.

Model Type: Matrix Factorization with implicit feedback
- Users represented as latent vectors (user_factors)
- Movies represented as latent vectors (movie_factors)
- Predicted score = dot product of user and movie vectors
- Loss function penalizes deviations from observed action_values

Training Data: interactions.parquet
- Positive signals: action_value > 0 (right=0.3, up=1.0)
- Negative signals: action_value = 0 (left swipes)

Output:
- user_factors.npy: (num_users, latent_dim) array
- movie_factors.npy: (num_movies, latent_dim) array
- model_metadata.json: training stats and hyperparameters

Algorithm: Alternating Least Squares (ALS) optimized for implicit feedback
- Faster than SGD for large datasets
- Alternates between fixing user factors and solving for movie factors, then vice versa
- Regularization prevents overfitting

Usage:
- Training: Given interactions, learn factors
- Inference: Score(user_id, movie_id) = user_factors[user_id] @ movie_factors[movie_id]
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import json
from typing import Tuple


class ImplicitALS:
    """
    Alternating Least Squares for Implicit Feedback.

    This implements the weighted matrix factorization approach where:
    - Observed interactions have confidence weights
    - Unobserved interactions are treated as weak negatives (not complete absence)

    Math:
    Minimize ||W * (R - U @ M.T)||^2 + lambda * (||U||^2 + ||M||^2)

    Where:
    - R: interaction matrix (num_users x num_movies) with action_values
    - W: confidence weights (higher for 'up', lower for 'left')
    - U: user factors (num_users x latent_dim)
    - M: movie factors (num_movies x latent_dim)
    - lambda: regularization strength
    """

    def __init__(self, latent_dim: int = 50, reg_lambda: float = 0.01, iterations: int = 20):
        """
        Args:
            latent_dim: Number of latent factors
            reg_lambda: L2 regularization strength
            iterations: Number of ALS iterations
        """
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.iterations = iterations

        self.user_factors = None
        self.movie_factors = None
        self.num_users = None
        self.num_movies = None

    def fit(self, interactions: pd.DataFrame, num_users: int, num_movies: int):
        """
        Train the model on interaction data.

        Args:
            interactions: DataFrame with columns [user_id, movie_id, action_value]
            num_users: Total number of users (determines matrix size)
            num_movies: Total number of movies (determines matrix size)

        Process:
        1. Build sparse interaction matrix
        2. Initialize factors randomly
        3. Alternate between updating user and movie factors
        4. Track loss for convergence monitoring
        """
        print(f"Initializing ALS with {num_users} users, {num_movies} movies, {self.latent_dim} factors...")

        self.num_users = num_users
        self.num_movies = num_movies

        # Build sparse interaction matrix
        # For each (user, movie) pair, use action_value as entry
        # Confidence weight: action_value itself works well (0, 0.3, 1.0)
        user_ids = interactions['user_id'].values
        movie_ids = interactions['movie_id'].values
        action_values = interactions['action_value'].values

        # Create sparse matrix
        interaction_matrix = coo_matrix(
            (action_values, (user_ids, movie_ids)),
            shape=(num_users, num_movies)
        ).tocsr()

        print(f"Interaction matrix: {interaction_matrix.shape}, {interaction_matrix.nnz} non-zero entries")

        # Initialize factors with small random values
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (num_users, self.latent_dim))
        self.movie_factors = np.random.normal(0, 0.01, (num_movies, self.latent_dim))

        # ALS iterations
        for iteration in range(self.iterations):
            print(f"\nIteration {iteration + 1}/{self.iterations}")

            # Fix movie factors, solve for user factors
            self.user_factors = self._solve_factors(
                interaction_matrix,
                self.movie_factors,
                self.reg_lambda
            )

            # Fix user factors, solve for movie factors
            self.movie_factors = self._solve_factors(
                interaction_matrix.T,
                self.user_factors,
                self.reg_lambda
            )

            # Compute loss (sampling subset for speed)
            if iteration % 5 == 0:
                loss = self._compute_loss(interaction_matrix)
                print(f"Loss: {loss:.4f}")

        print("\nTraining complete")

    def _solve_factors(self, interaction_matrix, fixed_factors, reg_lambda):
        """
        Solve for one set of factors while the other is fixed.

        For each row i in interaction_matrix:
        - Let r_i = observed values for row i
        - Let F = fixed_factors (other side of factorization)
        - Solve: argmin_x ||r_i - x @ F.T||^2 + lambda * ||x||^2

        Closed-form solution: x = (F.T @ F + lambda * I)^-1 @ F.T @ r_i

        Args:
            interaction_matrix: sparse matrix (N x M)
            fixed_factors: factors for the other side (M x latent_dim)
            reg_lambda: regularization

        Returns:
            new_factors: (N x latent_dim) array
        """
        N = interaction_matrix.shape[0]
        latent_dim = fixed_factors.shape[1]

        # Precompute F.T @ F (shared across all rows)
        FtF = fixed_factors.T @ fixed_factors
        reg_eye = reg_lambda * np.eye(latent_dim)

        new_factors = np.zeros((N, latent_dim))

        for i in range(N):
            # Get row i (observed interactions)
            row_data = interaction_matrix.getrow(i)

            if row_data.nnz == 0:
                # No interactions for this user/movie, use zero vector
                continue

            # Get indices and values
            indices = row_data.indices
            values = row_data.data

            # F_i: subset of fixed_factors for observed interactions
            F_i = fixed_factors[indices]

            # Solve: (F_i.T @ F_i + reg) @ x = F_i.T @ values
            A = F_i.T @ F_i + reg_eye
            b = F_i.T @ values

            new_factors[i] = np.linalg.solve(A, b)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{N} factors...", end='\r')

        print(f"  Processed {N}/{N} factors      ")
        return new_factors

    def _compute_loss(self, interaction_matrix, sample_size: int = 10000):
        """
        Compute reconstruction loss on a sample of interactions.

        Loss = sum of squared errors between observed and predicted values

        Args:
            interaction_matrix: sparse interaction matrix
            sample_size: number of interactions to sample for loss computation

        Returns:
            loss: float
        """
        # Sample random non-zero entries
        rows, cols = interaction_matrix.nonzero()
        sample_indices = np.random.choice(len(rows), size=min(sample_size, len(rows)), replace=False)

        sampled_rows = rows[sample_indices]
        sampled_cols = cols[sample_indices]
        true_values = np.array([interaction_matrix[r, c] for r, c in zip(sampled_rows, sampled_cols)])

        # Predict values
        predicted_values = np.sum(
            self.user_factors[sampled_rows] * self.movie_factors[sampled_cols],
            axis=1
        )

        # Mean squared error
        loss = np.mean((true_values - predicted_values) ** 2)
        return loss

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict score for a given user-movie pair.

        Args:
            user_id: int
            movie_id: int

        Returns:
            score: float (higher = better match)
        """
        return np.dot(self.user_factors[user_id], self.movie_factors[movie_id])

    def predict_batch(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """
        Predict scores for multiple user-movie pairs.

        Args:
            user_ids: array of user IDs
            movie_ids: array of movie IDs (same length as user_ids)

        Returns:
            scores: array of predicted scores
        """
        return np.sum(self.user_factors[user_ids] * self.movie_factors[movie_ids], axis=1)

    def recommend_for_user(self, user_id: int, top_k: int = 100, exclude_movie_ids: set = None) -> np.ndarray:
        """
        Generate top-k movie recommendations for a user.

        Args:
            user_id: int
            top_k: number of recommendations
            exclude_movie_ids: set of movie IDs to exclude (e.g., already seen)

        Returns:
            movie_ids: array of top-k movie IDs ranked by score
        """
        # Compute scores for all movies
        scores = self.user_factors[user_id] @ self.movie_factors.T

        # Exclude movies if specified
        if exclude_movie_ids:
            for mid in exclude_movie_ids:
                scores[mid] = -np.inf

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices

    def save(self, output_dir: Path):
        """
        Save trained model to disk.

        Saves:
        - user_factors.npy
        - movie_factors.npy
        - model_metadata.json
        """
        np.save(output_dir / 'user_factors.npy', self.user_factors)
        np.save(output_dir / 'movie_factors.npy', self.movie_factors)

        metadata = {
            'latent_dim': int(self.latent_dim),
            'reg_lambda': float(self.reg_lambda),
            'iterations': int(self.iterations),
            'num_users': int(self.num_users),
            'num_movies': int(self.num_movies)
        }

        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path):
        """
        Load trained model from disk.

        Args:
            model_dir: directory containing saved model files

        Returns:
            model: ImplicitALS instance
        """
        with open(model_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)

        model = cls(
            latent_dim=metadata['latent_dim'],
            reg_lambda=metadata['reg_lambda'],
            iterations=metadata['iterations']
        )

        model.user_factors = np.load(model_dir / 'user_factors.npy')
        model.movie_factors = np.load(model_dir / 'movie_factors.npy')
        model.num_users = metadata['num_users']
        model.num_movies = metadata['num_movies']

        print(f"Model loaded from {model_dir}")
        return model


def main(interactions_path: Path, output_dir: Path, latent_dim: int = 50, iterations: int = 20):
    """
    Train collaborative filtering model on interaction data.

    Args:
        interactions_path: Path to interactions.parquet
        output_dir: Directory to save trained model
        latent_dim: Number of latent factors
        iterations: Number of ALS iterations
    """
    print(f"Loading interactions from {interactions_path}...")
    interactions = pd.read_parquet(interactions_path)
    print(f"Loaded {len(interactions)} interactions")

    # Determine matrix dimensions
    num_users = interactions['user_id'].max() + 1
    num_movies = interactions['movie_id'].max() + 1
    print(f"Matrix size: {num_users} users x {num_movies} movies")

    # Initialize and train model
    model = ImplicitALS(latent_dim=latent_dim, reg_lambda=0.01, iterations=iterations)
    model.fit(interactions, num_users, num_movies)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    # Test: recommend for a sample user
    print("\nTest: Recommendations for user 0:")
    top_movies = model.recommend_for_user(0, top_k=10)
    print(f"Top 10 movie IDs: {top_movies}")

    return model


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python collaborative_filtering.py <interactions_parquet> <output_dir> [latent_dim] [iterations]")
        print("Example: python collaborative_filtering.py ../../output/processed/interactions.parquet ../../output/models 50 20")
        sys.exit(1)

    interactions_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    latent_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    main(interactions_path, output_dir, latent_dim, iterations)
