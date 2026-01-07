"""
TMDB DATA FETCHER MODULE

Purpose: Fetch movie data from TMDb API using IMDb IDs.

Input: Parquet file with tconst (IMDb IDs)
Output: Parquet file with TMDb enrichment data

TMDb fields fetched:
- tmdb_id: TMDb internal ID
- tmdb_popularity: TMDb popularity score
- tmdb_rating: TMDb vote_average
- tmdb_vote_count: TMDb vote count
- description: Movie overview/plot summary
- poster_path: Poster image path
- backdrop_path: Backdrop image path
- tmdb_genres: TMDb genre names
- keywords: TMDb keywords
- budget: Production budget
- revenue: Box office revenue

Rate Limiting:
- TMDb allows 40 requests/second
- This script implements conservative rate limiting (10 req/sec)
- Progress is saved every 100 movies for resume capability
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import json


class TMDbFetcher:
    """Handles TMDb API requests with rate limiting and caching."""

    def __init__(self, api_key: str):
        """
        Initialize TMDb fetcher.

        Args:
            api_key: TMDb API key
        """
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()

        # Rate limiting: 10 requests per second (conservative)
        self.min_request_interval = 0.1  # 100ms between requests
        self.last_request_time = 0

        # Get image configuration
        self.img_config = self._get_image_config()

    def _get_image_config(self) -> Dict[str, Any]:
        """Fetch TMDb image configuration."""
        url = f"{self.base_url}/configuration?api_key={self.api_key}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()["images"]

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def find_by_imdb_id(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Find TMDb movie by IMDb ID and fetch detailed info.

        Args:
            imdb_id: IMDb tconst (e.g., 'tt0111161')

        Returns:
            Dictionary with TMDb data, or None if not found
        """
        self._rate_limit()

        try:
            # Step 1: Find TMDb ID using IMDb ID
            find_url = f"{self.base_url}/find/{imdb_id}"
            find_params = {
                "api_key": self.api_key,
                "external_source": "imdb_id"
            }

            find_response = self.session.get(find_url, params=find_params)
            find_response.raise_for_status()
            find_data = find_response.json()

            # Check if movie was found
            movie_results = find_data.get("movie_results", [])
            if not movie_results:
                return None

            tmdb_id = movie_results[0]["id"]

            # Step 2: Get detailed movie info
            self._rate_limit()

            movie_url = f"{self.base_url}/movie/{tmdb_id}"
            movie_params = {
                "api_key": self.api_key,
                "append_to_response": "credits,keywords"
            }

            movie_response = self.session.get(movie_url, params=movie_params)
            movie_response.raise_for_status()
            movie_data = movie_response.json()

            # Extract and format data
            base_url = self.img_config["secure_base_url"]
            poster_size = "w500"

            return {
                "tconst": imdb_id,
                "tmdb_id": tmdb_id,
                "tmdb_popularity": movie_data.get("popularity"),
                "tmdb_rating": movie_data.get("vote_average"),
                "tmdb_vote_count": movie_data.get("vote_count"),
                "description": movie_data.get("overview"),
                "poster_path": movie_data.get("poster_path"),
                "backdrop_path": movie_data.get("backdrop_path"),
                "poster_url": (
                    f"{base_url}{poster_size}{movie_data['poster_path']}"
                    if movie_data.get("poster_path") else None
                ),
                "backdrop_url": (
                    f"{base_url}{poster_size}{movie_data['backdrop_path']}"
                    if movie_data.get("backdrop_path") else None
                ),
                "tmdb_genres": [g["name"] for g in movie_data.get("genres", [])],
                "keywords": [k["name"] for k in movie_data.get("keywords", {}).get("keywords", [])],
                "budget": movie_data.get("budget"),
                "revenue": movie_data.get("revenue"),
                "runtime_tmdb": movie_data.get("runtime"),
                "production_countries": [
                    c["iso_3166_1"] for c in movie_data.get("production_countries", [])
                ],
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {imdb_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for {imdb_id}: {e}")
            return None


def fetch_tmdb_for_movies(
    movies_df: pd.DataFrame,
    api_key: str,
    cache_file: Optional[Path] = None,
    max_movies: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch TMDb data for movies in dataframe.

    Args:
        movies_df: DataFrame with 'tconst' column (IMDb IDs)
        api_key: TMDb API key
        cache_file: Optional path to save/load progress
        max_movies: Optional limit on number of movies to process

    Returns:
        DataFrame with TMDb enrichment data
    """
    fetcher = TMDbFetcher(api_key)

    # Load existing cache if available
    if cache_file and cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        cached_df = pd.read_parquet(cache_file)
        already_fetched = set(cached_df['tconst'].values)
        tmdb_data = cached_df.to_dict('records')
    else:
        already_fetched = set()
        tmdb_data = []

    # Get movies to fetch
    movies_to_fetch = movies_df[~movies_df['tconst'].isin(already_fetched)]

    if max_movies:
        movies_to_fetch = movies_to_fetch.head(max_movies)

    total = len(movies_to_fetch)
    print(f"Fetching TMDb data for {total} movies...")
    print(f"Already cached: {len(already_fetched)} movies")

    # Fetch data
    success_count = 0
    fail_count = 0

    for idx, (_, row) in enumerate(movies_to_fetch.iterrows(), 1):
        imdb_id = row['tconst']

        # Progress update every 10 movies
        if idx % 10 == 0:
            print(f"Progress: {idx}/{total} ({success_count} found, {fail_count} not found)")

        # Fetch TMDb data
        movie_data = fetcher.find_by_imdb_id(imdb_id)

        if movie_data:
            tmdb_data.append(movie_data)
            success_count += 1
        else:
            fail_count += 1

        # Save progress every 100 movies
        if cache_file and idx % 100 == 0:
            temp_df = pd.DataFrame(tmdb_data)
            temp_df.to_parquet(cache_file, index=False)
            print(f"Saved progress to {cache_file}")

    print(f"\nCompleted: {success_count} successful, {fail_count} failed")

    # Create final dataframe
    tmdb_df = pd.DataFrame(tmdb_data)

    # Save final cache
    if cache_file:
        tmdb_df.to_parquet(cache_file, index=False)
        print(f"Final data saved to {cache_file}")

    return tmdb_df


def main(movies_parquet_path: Path, output_dir: Path, api_key: str, max_movies: Optional[int] = None):
    """
    Main execution: Fetch TMDb data for all movies.

    Args:
        movies_parquet_path: Path to movies parquet from data_clean.py
        output_dir: Directory to save TMDb data
        api_key: TMDb API key
        max_movies: Optional limit for testing
    """
    print("Loading movies...")
    movies_df = pd.read_parquet(movies_parquet_path)
    print(f"Loaded {len(movies_df)} movies")

    # Setup cache file
    cache_file = output_dir / 'tmdb_enrichment.parquet'

    # Fetch TMDb data
    tmdb_df = fetch_tmdb_for_movies(
        movies_df,
        api_key,
        cache_file=cache_file,
        max_movies=max_movies
    )

    print(f"\nFinal TMDb dataset: {len(tmdb_df)} movies")
    print(f"Sample record:")
    if len(tmdb_df) > 0:
        print(tmdb_df.iloc[0].to_dict())


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("Usage: python fetch_tmdb_data.py <movies_parquet> <output_dir> <api_key> [max_movies]")
        print("Example: python fetch_tmdb_data.py ../../output/movies_core.parquet ../../output a55b214aa396861a2625258556bbc6ee 100")
        sys.exit(1)

    movies_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    api_key = sys.argv[3]
    max_movies = int(sys.argv[4]) if len(sys.argv) > 4 else None

    output_dir.mkdir(parents=True, exist_ok=True)

    main(movies_path, output_dir, api_key, max_movies)
