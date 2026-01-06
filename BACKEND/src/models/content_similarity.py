"""
CONTENT-BASED SIMILARITY MODULE

Purpose: Enable semantic similarity matching between movies using:
- TMDb plot descriptions (overview text)
- TMDb keywords (tags like "time travel", "revenge", etc.)

This allows recommendations based on movie content/themes rather than just
collaborative filtering or genre matching.

Algorithm:
1. Combine description + keywords into single text corpus per movie
2. Build TF-IDF (Term Frequency-Inverse Document Frequency) matrix
3. Compute cosine similarity between all movie pairs
4. Store similarity matrix for fast lookups

Usage:
    content_sim = ContentSimilarity(movies_df)
    similar_movies = content_sim.get_similar_movies(movie_id=42, top_k=20)

    # Or get similarity between specific movies
    similarity_score = content_sim.get_similarity(movie_id1=42, movie_id2=100)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ContentSimilarity:
    """
    Compute and store content-based similarity between movies.
    Uses TF-IDF on descriptions + keywords.
    """

    def __init__(self, movies_df: pd.DataFrame, precompute_all: bool = False):
        """
        Initialize content similarity engine.

        Args:
            movies_df: DataFrame with columns: movie_id, description, keywords
            precompute_all: If True, compute full similarity matrix upfront (memory intensive)
        """
        self.movies_df = movies_df

        # Build text corpus for each movie
        print("Building text corpus from descriptions and keywords...")
        self.corpus = self._build_corpus(movies_df)

        # Build TF-IDF matrix
        print("Computing TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,        # Limit vocabulary size
            min_df=2,                 # Ignore terms that appear in < 2 movies
            max_df=0.8,               # Ignore terms that appear in > 80% of movies
            ngram_range=(1, 2),       # Use unigrams and bigrams
            stop_words='english'      # Remove common English words
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Optionally precompute full similarity matrix
        self.similarity_matrix = None
        if precompute_all:
            print("Precomputing full similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            print(f"Similarity matrix shape: {self.similarity_matrix.shape}")

        # Create movie_id to index mapping
        self.movie_id_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(movies_df['movie_id'])
        }
        self.idx_to_movie_id = {
            idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()
        }

    def _build_corpus(self, movies_df: pd.DataFrame) -> List[str]:
        """
        Build text corpus by combining description + keywords for each movie.

        Args:
            movies_df: DataFrame with description and keywords columns

        Returns:
            List of text documents (one per movie)
        """
        corpus = []

        for _, row in movies_df.iterrows():
            text_parts = []

            # Add description (if available)
            if pd.notna(row['description']) and row['description']:
                text_parts.append(str(row['description']))

            # Add keywords (if available)
            keywords = row['keywords']
            if isinstance(keywords, (list, np.ndarray)) and len(keywords) > 0:
                # Repeat keywords to give them more weight in TF-IDF
                keywords_text = ' '.join(str(k) for k in keywords) + ' ' + ' '.join(str(k) for k in keywords)
                text_parts.append(keywords_text)

            # Combine all text
            if text_parts:
                corpus.append(' '.join(text_parts))
            else:
                # Empty document for movies without content data
                corpus.append('')

        return corpus

    def get_similar_movies(
        self,
        movie_id: int,
        top_k: int = 20,
        min_similarity: float = 0.05
    ) -> List[Tuple[int, float]]:
        """
        Get most similar movies to given movie based on content.

        Args:
            movie_id: Target movie ID
            top_k: Number of similar movies to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (movie_id, similarity_score) tuples, sorted by score descending
        """
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]

        # Compute similarities
        if self.similarity_matrix is not None:
            # Use precomputed matrix
            similarities = self.similarity_matrix[idx]
        else:
            # Compute on-demand
            movie_vector = self.tfidf_matrix[idx]
            similarities = cosine_similarity(movie_vector, self.tfidf_matrix)[0]

        # Get top-k similar movies (excluding self)
        similar_indices = np.argsort(similarities)[::-1]

        results = []
        for sim_idx in similar_indices:
            if sim_idx == idx:
                continue  # Skip self

            similarity_score = similarities[sim_idx]
            if similarity_score < min_similarity:
                break

            similar_movie_id = self.idx_to_movie_id[sim_idx]
            results.append((similar_movie_id, similarity_score))

            if len(results) >= top_k:
                break

        return results

    def get_similarity(self, movie_id1: int, movie_id2: int) -> float:
        """
        Get similarity score between two specific movies.

        Args:
            movie_id1: First movie ID
            movie_id2: Second movie ID

        Returns:
            Similarity score (0-1)
        """
        if movie_id1 not in self.movie_id_to_idx or movie_id2 not in self.movie_id_to_idx:
            return 0.0

        idx1 = self.movie_id_to_idx[movie_id1]
        idx2 = self.movie_id_to_idx[movie_id2]

        if self.similarity_matrix is not None:
            return self.similarity_matrix[idx1, idx2]
        else:
            vec1 = self.tfidf_matrix[idx1]
            vec2 = self.tfidf_matrix[idx2]
            return cosine_similarity(vec1, vec2)[0, 0]

    def get_batch_similarity(
        self,
        target_movie_id: int,
        candidate_movie_ids: List[int]
    ) -> np.ndarray:
        """
        Get similarity scores between target movie and multiple candidates.
        Optimized for batch processing in recommendation engine.

        Args:
            target_movie_id: Target movie ID
            candidate_movie_ids: List of candidate movie IDs

        Returns:
            Array of similarity scores (same order as candidate_movie_ids)
        """
        if target_movie_id not in self.movie_id_to_idx:
            return np.zeros(len(candidate_movie_ids))

        target_idx = self.movie_id_to_idx[target_movie_id]
        target_vector = self.tfidf_matrix[target_idx]

        # Build matrix of candidate vectors
        candidate_indices = [
            self.movie_id_to_idx[mid] for mid in candidate_movie_ids
            if mid in self.movie_id_to_idx
        ]

        if not candidate_indices:
            return np.zeros(len(candidate_movie_ids))

        candidate_matrix = self.tfidf_matrix[candidate_indices]

        # Compute similarities
        similarities = cosine_similarity(target_vector, candidate_matrix)[0]

        # Map back to original candidate order
        result = np.zeros(len(candidate_movie_ids))
        valid_idx = 0
        for i, mid in enumerate(candidate_movie_ids):
            if mid in self.movie_id_to_idx:
                result[i] = similarities[valid_idx]
                valid_idx += 1

        return result

    def save(self, output_dir: Path):
        """Save model to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save TF-IDF matrix and vectorizer
        with open(output_dir / 'content_similarity.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id,
                'similarity_matrix': self.similarity_matrix
            }, f)

        print(f"Content similarity model saved to {output_dir}")

    @classmethod
    def load(cls, models_dir: Path):
        """Load model from disk."""
        models_dir = Path(models_dir)

        with open(models_dir / 'content_similarity.pkl', 'rb') as f:
            data = pickle.load(f)

        # Create instance without running __init__
        instance = cls.__new__(cls)
        instance.vectorizer = data['vectorizer']
        instance.tfidf_matrix = data['tfidf_matrix']
        instance.movie_id_to_idx = data['movie_id_to_idx']
        instance.idx_to_movie_id = data['idx_to_movie_id']
        instance.similarity_matrix = data['similarity_matrix']
        instance.movies_df = None
        instance.corpus = None

        print(f"Content similarity model loaded from {models_dir}")
        return instance


def train_content_similarity(movies_path: Path, output_dir: Path):
    """
    Train content similarity model and save to disk.

    Args:
        movies_path: Path to movies.parquet with TMDb data
        output_dir: Directory to save model
    """
    print("Loading movies...")
    movies_df = pd.read_parquet(movies_path)

    print(f"Total movies: {len(movies_df)}")
    movies_with_content = movies_df[
        movies_df['description'].notna() | movies_df['keywords'].notna()
    ]
    print(f"Movies with content data: {len(movies_with_content)}")

    print("\nTraining content similarity model...")
    content_sim = ContentSimilarity(movies_df, precompute_all=False)

    print("\nSaving model...")
    content_sim.save(output_dir)

    # Test with sample movie
    print("\n=== TEST: Similar movies ===")
    sample_movie = movies_df[movies_df['description'].notna()].iloc[0]
    print(f"\nBase movie: {sample_movie['title']} ({sample_movie['year']})")
    print(f"Genres: {sample_movie['genres']}")

    similar = content_sim.get_similar_movies(sample_movie['movie_id'], top_k=5)
    print(f"\nTop 5 similar movies by content:")
    for rank, (movie_id, score) in enumerate(similar, 1):
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"{rank}. {movie['title']} ({movie['year']}) - Similarity: {score:.3f}")
        print(f"   Genres: {movie['genres']}")

    print("\nContent similarity model training complete!")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python content_similarity.py <movies_parquet> <output_dir>")
        print("Example: python content_similarity.py ../../output/processed/movies.parquet ../../output/models")
        sys.exit(1)

    movies_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    train_content_similarity(movies_path, output_dir)
