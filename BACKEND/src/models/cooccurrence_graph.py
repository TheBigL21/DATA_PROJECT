"""
MOVIE CO-OCCURRENCE GRAPH

Purpose: Build a graph capturing which movies are related based on session behavior.

Concept:
- Two movies are "related" if users interact positively with both in same session
- Edge weight = how often movies co-occur in positive contexts
- Used for neighborhood-based recommendations: "users who liked X also liked Y"

Graph Structure:
- Nodes: movies (identified by movie_id)
- Edges: weighted connections between movies
- Edge weight formula:
  * Increment when both movies get 'right' or 'up' in same session
  * Decrement (slightly) when one gets 'up' but other gets 'left' (anti-correlation)

Storage: Sparse adjacency matrix (num_movies x num_movies)
- Only store edges with weight > threshold
- Symmetric matrix (undirected graph)

Usage:
- Given movie M that user just swiped 'right' on
- Retrieve top-k neighbors of M from graph
- Recommend those neighbors in next swipes
- This creates "exploration around interest" behavior
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import List, Tuple
import json


class MovieCooccurrenceGraph:
    """
    Build and query movie-movie co-occurrence graph.

    Construction:
    - Process each session
    - For movies with positive signals (right/up):
      - Create edges between all pairs
      - Weight based on signal strength (up-up > right-right > right-up)
    - For movies with mixed signals:
      - Negative edges between 'left' and 'right/up' movies (weak)

    Query:
    - Get neighbors of movie M
    - Return top-k most strongly connected movies
    """

    def __init__(self, num_movies: int):
        """
        Args:
            num_movies: Total number of movies in the system
        """
        self.num_movies = num_movies
        # Adjacency matrix (sparse)
        self.adjacency = csr_matrix((num_movies, num_movies), dtype=np.float32)

    def build_from_interactions(self, interactions: pd.DataFrame):
        """
        Build graph from interaction data.

        Process:
        1. Group interactions by session
        2. For each session:
           - Extract movies with positive signals
           - Create co-occurrence edges between them
           - Weight edges by signal strength product

        Args:
            interactions: DataFrame with [session_id, movie_id, action, action_value]
        """
        print(f"Building co-occurrence graph for {self.num_movies} movies...")

        # Initialize edge accumulator (use dict for efficiency, convert to sparse later)
        edge_weights = {}

        # Group by session
        sessions = interactions.groupby('session_id')
        num_sessions = len(sessions)

        for session_idx, (session_id, session_data) in enumerate(sessions):
            if (session_idx + 1) % 1000 == 0:
                print(f"  Processed {session_idx + 1}/{num_sessions} sessions...", end='\r')

            # Separate movies by action
            positive_movies = session_data[session_data['action'].isin(['right', 'up'])]
            negative_movies = session_data[session_data['action'] == 'left']

            # Build edges between positive movies
            for i, row_i in positive_movies.iterrows():
                movie_i = row_i['movie_id']
                signal_i = row_i['action_value']

                for j, row_j in positive_movies.iterrows():
                    if i >= j:
                        continue  # Avoid self-loops and duplicates

                    movie_j = row_j['movie_id']
                    signal_j = row_j['action_value']

                    # Edge weight = product of signals
                    # up-up: 1.0 * 1.0 = 1.0 (strongest)
                    # up-right: 1.0 * 0.3 = 0.3
                    # right-right: 0.3 * 0.3 = 0.09
                    weight = signal_i * signal_j

                    # Symmetric edge
                    edge_key_1 = (min(movie_i, movie_j), max(movie_i, movie_j))
                    edge_weights[edge_key_1] = edge_weights.get(edge_key_1, 0.0) + weight

            # Optional: negative edges between positive and negative movies (weak signal)
            # This captures "user liked A but disliked B, so A and B are dissimilar"
            for _, row_pos in positive_movies.iterrows():
                movie_pos = row_pos['movie_id']

                for _, row_neg in negative_movies.iterrows():
                    movie_neg = row_neg['movie_id']

                    # Small negative weight
                    weight = -0.05

                    edge_key = (min(movie_pos, movie_neg), max(movie_pos, movie_neg))
                    edge_weights[edge_key] = edge_weights.get(edge_key, 0.0) + weight

        print(f"  Processed {num_sessions}/{num_sessions} sessions      ")
        print(f"Total edges created: {len(edge_weights)}")

        # Convert edge_weights dict to sparse matrix
        rows = []
        cols = []
        weights = []

        for (i, j), w in edge_weights.items():
            # Symmetric: add both (i,j) and (j,i)
            rows.append(i)
            cols.append(j)
            weights.append(w)

            rows.append(j)
            cols.append(i)
            weights.append(w)

        self.adjacency = csr_matrix(
            (weights, (rows, cols)),
            shape=(self.num_movies, self.num_movies),
            dtype=np.float32
        )

        print(f"Adjacency matrix: {self.adjacency.shape}, {self.adjacency.nnz} non-zero entries")

    def get_neighbors(self, movie_id: int, top_k: int = 50, min_weight: float = 0.1) -> List[Tuple[int, float]]:
        """
        Get top-k most similar movies to a given movie.

        Args:
            movie_id: int
            top_k: number of neighbors to return
            min_weight: minimum edge weight threshold

        Returns:
            List of (neighbor_movie_id, weight) tuples, sorted by weight descending
        """
        # Extract row for this movie
        row = self.adjacency.getrow(movie_id)

        # Get non-zero entries
        neighbors = []
        for idx, weight in zip(row.indices, row.data):
            if weight >= min_weight:
                neighbors.append((idx, weight))

        # Sort by weight descending
        neighbors.sort(key=lambda x: x[1], reverse=True)

        return neighbors[:top_k]

    def get_neighbors_batch(self, movie_ids: List[int], top_k: int = 50, min_weight: float = 0.1) -> List[Tuple[int, float]]:
        """
        Get neighbors for multiple movies and aggregate.

        Use case: User swiped right on multiple movies in session, get combined neighborhood.

        Args:
            movie_ids: list of movie IDs
            top_k: total neighbors to return
            min_weight: minimum edge weight

        Returns:
            List of (movie_id, aggregated_weight) sorted by weight
        """
        # Accumulate neighbor weights
        neighbor_weights = {}

        for movie_id in movie_ids:
            neighbors = self.get_neighbors(movie_id, top_k=100, min_weight=min_weight)

            for neighbor_id, weight in neighbors:
                # Exclude input movies
                if neighbor_id in movie_ids:
                    continue

                neighbor_weights[neighbor_id] = neighbor_weights.get(neighbor_id, 0.0) + weight

        # Sort by aggregated weight
        sorted_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)

        return sorted_neighbors[:top_k]

    def save(self, output_dir: Path):
        """
        Save graph to disk.

        Saves:
        - adjacency_matrix.npz (sparse matrix)
        - graph_metadata.json
        """
        save_npz(output_dir / 'adjacency_matrix.npz', self.adjacency)

        metadata = {
            'num_movies': int(self.num_movies),
            'num_edges': int(self.adjacency.nnz),
        }

        with open(output_dir / 'graph_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Graph saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path):
        """
        Load graph from disk.

        Args:
            model_dir: directory containing saved graph files

        Returns:
            graph: MovieCooccurrenceGraph instance
        """
        with open(model_dir / 'graph_metadata.json', 'r') as f:
            metadata = json.load(f)

        graph = cls(num_movies=metadata['num_movies'])
        graph.adjacency = load_npz(model_dir / 'adjacency_matrix.npz')

        print(f"Graph loaded from {model_dir}")
        return graph


def main(interactions_path: Path, output_dir: Path):
    """
    Build co-occurrence graph from interaction data.

    Args:
        interactions_path: Path to interactions.parquet
        output_dir: Directory to save graph
    """
    print(f"Loading interactions from {interactions_path}...")
    interactions = pd.read_parquet(interactions_path)
    print(f"Loaded {len(interactions)} interactions")

    # Determine number of movies
    num_movies = interactions['movie_id'].max() + 1
    print(f"Number of movies: {num_movies}")

    # Build graph
    graph = MovieCooccurrenceGraph(num_movies)
    graph.build_from_interactions(interactions)

    # Save graph
    output_dir.mkdir(parents=True, exist_ok=True)
    graph.save(output_dir)

    # Test: get neighbors for a sample movie
    print("\nTest: Neighbors of movie 100:")
    neighbors = graph.get_neighbors(100, top_k=10)
    print(f"Top 10 neighbors: {neighbors}")

    return graph


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python cooccurrence_graph.py <interactions_parquet> <output_dir>")
        print("Example: python cooccurrence_graph.py ../../output/processed/interactions.parquet ../../output/models")
        sys.exit(1)

    interactions_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    main(interactions_path, output_dir)
