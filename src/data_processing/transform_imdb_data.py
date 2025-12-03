"""
DATA TRANSFORMATION MODULE

Purpose: Convert IMDb parquet files into internal app schema.

Input: IMDb parquet files from data_clean.py:
  - movies_core.parquet
  - movie_genres.parquet
  - movie_ratings.parquet
  - movie_people_links.parquet
  - people_core.parquet

Output: Internal app schema:
  - movies.parquet: (movie_id, title, year, runtime, genres_list, avg_rating, num_votes, director, top_actors)

Logic:
1. Load all IMDb tables
2. Join genres into list per movie
3. Join ratings
4. Extract director name and top 5 actor names
5. Create integer movie_id (for ML model efficiency)
6. Output single denormalized movies table optimized for recommendation engine
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_imdb_tables(imdb_output_dir: Path) -> dict:
    """
    Load all cleaned IMDb parquet files.

    Returns dict with keys: movies_core, movie_genres, movie_ratings, movie_people_links, people_core, country_mapping, tmdb_enrichment
    """
    tables = {}
    tables['movies_core'] = pd.read_parquet(imdb_output_dir / 'movies_core.parquet')
    tables['movie_genres'] = pd.read_parquet(imdb_output_dir / 'movie_genres.parquet')
    tables['movie_ratings'] = pd.read_parquet(imdb_output_dir / 'movie_ratings.parquet')
    tables['movie_people_links'] = pd.read_parquet(imdb_output_dir / 'movie_people_links.parquet')
    tables['people_core'] = pd.read_parquet(imdb_output_dir / 'people_core.parquet')

    # Load country mapping if available
    country_path = imdb_output_dir / 'country_mapping.parquet'
    if country_path.exists():
        tables['country_mapping'] = pd.read_parquet(country_path)
        print(f"Loaded {len(tables['country_mapping'])} country mappings")
    else:
        tables['country_mapping'] = None
        print("Warning: country_mapping.parquet not found, movies will have 'Unknown' country")

    # Load TMDb enrichment if available
    tmdb_path = imdb_output_dir / 'tmdb_enrichment.parquet'
    if tmdb_path.exists():
        tables['tmdb_enrichment'] = pd.read_parquet(tmdb_path)
        print(f"Loaded TMDb data for {len(tables['tmdb_enrichment'])} movies")
    else:
        tables['tmdb_enrichment'] = None
        print("Warning: tmdb_enrichment.parquet not found, movies will not have TMDb data")

    return tables


def build_genre_lists(movies_core: pd.DataFrame, movie_genres: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate genres into list per movie.

    Input: movie_genres has one row per (tconst, genre)
    Output: DataFrame with (tconst, genres) where genres is a list

    Example: tconst='tt0111161' -> genres=['crime', 'drama']
    """
    # Group by tconst and aggregate genres into list
    genre_lists = (
        movie_genres
        .groupby('tconst')['genre']
        .apply(list)
        .reset_index()
        .rename(columns={'genre': 'genres'})
    )

    # Merge with movies_core to ensure all movies have genre list
    movies_with_genres = movies_core.merge(genre_lists, on='tconst', how='left')

    # Fill missing genres with empty list (should not happen due to data_clean.py constraints)
    movies_with_genres['genres'] = movies_with_genres['genres'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return movies_with_genres


def extract_cast_crew(movie_people_links: pd.DataFrame, people_core: pd.DataFrame) -> pd.DataFrame:
    """
    Extract director name and top 5 actor names per movie.

    Logic:
    - For directors: take first director by ordering (usually only one)
    - For actors: take top 5 by ordering (already filtered in data_clean.py)

    Output: DataFrame with (tconst, director, actors) where actors is list of names
    """
    # Merge people names into movie_people_links
    links_with_names = movie_people_links.merge(
        people_core[['nconst', 'primaryName']],
        on='nconst',
        how='left'
    )

    # Extract directors (take first one by ordering)
    directors = (
        links_with_names[links_with_names['category'] == 'director']
        .sort_values(['tconst', 'ordering'])
        .groupby('tconst')
        .first()
        .reset_index()[['tconst', 'primaryName']]
        .rename(columns={'primaryName': 'director'})
    )

    # Extract actors (top 5, already limited in data_clean.py)
    actors = (
        links_with_names[links_with_names['category'].isin(['actor', 'actress'])]
        .sort_values(['tconst', 'ordering'])
        .groupby('tconst')['primaryName']
        .apply(list)
        .reset_index()
        .rename(columns={'primaryName': 'actors'})
    )

    # Merge directors and actors
    cast_crew = directors.merge(actors, on='tconst', how='left')

    # Fill missing actors with empty list
    cast_crew['actors'] = cast_crew['actors'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return cast_crew


def build_internal_movies_table(tables: dict) -> pd.DataFrame:
    """
    Construct final internal movies table by joining all components.

    Steps:
    1. Start with movies_core
    2. Add genre lists
    3. Add ratings
    4. Add cast/crew
    5. Add country info
    6. Create integer movie_id
    7. Select and rename columns to match app schema

    Output schema:
    - movie_id: int (0, 1, 2, ...)
    - tconst: str (original IMDb ID for reference)
    - title: str (primaryTitle)
    - year: int
    - runtime: int (minutes)
    - genres: list[str]
    - avg_rating: float (IMDb rating)
    - num_votes: int (IMDb votes)
    - composite_rating: float (Combined IMDb + TMDb rating: 60% IMDb + 40% TMDb)
    - director: str
    - actors: list[str] (top 5)
    - country: str (production country)
    - tmdb_id: int (TMDb ID)
    - tmdb_popularity: float (TMDb popularity score)
    - tmdb_rating: float (TMDb vote_average)
    - tmdb_vote_count: int (TMDb vote count)
    - description: str (TMDb overview/plot)
    - poster_path: str (TMDb poster path)
    - backdrop_path: str (TMDb backdrop path)
    - poster_url: str (Full TMDb poster URL)
    - backdrop_url: str (Full TMDb backdrop URL)
    - keywords: list[str] (TMDb keywords)
    - budget: int (Production budget)
    - revenue: int (Box office revenue)
    """
    # Step 1: Build genre lists
    movies = build_genre_lists(tables['movies_core'], tables['movie_genres'])

    # Step 2: Add ratings
    movies = movies.merge(tables['movie_ratings'], on='tconst', how='inner')

    # Step 3: Add cast/crew
    cast_crew = extract_cast_crew(tables['movie_people_links'], tables['people_core'])
    movies = movies.merge(cast_crew, on='tconst', how='left')

    # Step 4: Add country data (if available)
    if 'country_mapping' in tables and tables['country_mapping'] is not None:
        movies = movies.merge(tables['country_mapping'], on='tconst', how='left')
        movies['country'] = movies['country'].fillna('Unknown')
    else:
        movies['country'] = 'Unknown'

    # Step 4.5: Add TMDb enrichment data (if available)
    if 'tmdb_enrichment' in tables and tables['tmdb_enrichment'] is not None:
        tmdb_cols = ['tconst', 'tmdb_id', 'tmdb_popularity', 'tmdb_rating', 'tmdb_vote_count',
                     'description', 'poster_path', 'backdrop_path', 'poster_url', 'backdrop_url',
                     'tmdb_genres', 'keywords', 'budget', 'revenue']
        tmdb_data = tables['tmdb_enrichment'][tmdb_cols]
        movies = movies.merge(tmdb_data, on='tconst', how='left')
        print(f"Merged TMDb data: {movies['tmdb_id'].notna().sum()} movies have TMDb info")

        # Step 4.6: Merge TMDB genres with IMDb genres
        # Strategy: Combine both genre lists and remove duplicates
        def merge_genres(imdb_genres, tmdb_genres):
            """Merge IMDb and TMDB genre lists, removing duplicates (case-insensitive)"""
            if not isinstance(imdb_genres, list):
                imdb_genres = []

            # Convert tmdb_genres to list if it's a numpy array or other iterable
            if tmdb_genres is None or (hasattr(tmdb_genres, '__len__') and len(tmdb_genres) == 0):
                tmdb_genres = []
            elif not isinstance(tmdb_genres, (list, np.ndarray)):
                tmdb_genres = []
            else:
                # Convert to list if it's an array
                tmdb_genres = list(tmdb_genres)

            # Normalize TMDB genres to lowercase
            try:
                tmdb_genres_lower = [str(g).lower() for g in tmdb_genres if g and str(g).strip()]
            except:
                tmdb_genres_lower = []

            # Combine and deduplicate
            combined = list(imdb_genres)  # Start with IMDb genres
            for genre in tmdb_genres_lower:
                if genre not in combined:
                    combined.append(genre)

            return combined

        movies['genres'] = movies.apply(lambda row: merge_genres(row['genres'], row['tmdb_genres']), axis=1)
        print(f"Merged TMDB genres into main genres field")
    else:
        # Add empty TMDb columns if not available
        movies['tmdb_id'] = None
        movies['tmdb_popularity'] = None
        movies['tmdb_rating'] = None
        movies['tmdb_vote_count'] = None
        movies['description'] = None
        movies['poster_path'] = None
        movies['backdrop_path'] = None
        movies['poster_url'] = None
        movies['backdrop_url'] = None
        movies['keywords'] = None
        movies['budget'] = None
        movies['revenue'] = None

    # Fill missing director with 'Unknown'
    movies['director'] = movies['director'].fillna('Unknown')
    movies['actors'] = movies['actors'].apply(lambda x: x if isinstance(x, list) else [])

    # Step 5: Create integer movie_id (0-indexed)
    movies = movies.reset_index(drop=True)
    movies['movie_id'] = movies.index

    # Step 6: Rename columns FIRST
    movies = movies.rename(columns={
        'primaryTitle': 'title',
        'startYear': 'year',
        'runtimeMinutes': 'runtime',
        'averageRating': 'avg_rating',
        'numVotes': 'num_votes'
    })

    # Step 7: Calculate composite rating (IMDb + TMDb combined)
    # New system: Weighted average based on vote reliability + normalized to 0-100
    def calculate_composite_rating(row):
        imdb_rating = row['avg_rating']  # 0-10 scale
        imdb_votes = row['num_votes']
        tmdb_rating = row['tmdb_rating']  # 0-10 scale
        tmdb_votes = row['tmdb_vote_count']

        # If TMDB data is available, use weighted average based on vote counts
        if pd.notna(tmdb_rating) and pd.notna(tmdb_votes) and tmdb_rating > 0 and tmdb_votes > 0:
            # Calculate weights based on vote counts (more votes = more reliable)
            # Use log scale to prevent extreme dominance
            imdb_weight = np.log1p(imdb_votes)
            tmdb_weight = np.log1p(tmdb_votes)

            total_weight = imdb_weight + tmdb_weight

            # Weighted average (still on 0-10 scale)
            if total_weight > 0:
                composite = (imdb_rating * imdb_weight + tmdb_rating * tmdb_weight) / total_weight
            else:
                composite = imdb_rating
        else:
            # Only IMDb rating available
            composite = imdb_rating

        return composite

    movies['composite_rating'] = movies.apply(calculate_composite_rating, axis=1)

    # Step 7.5: Add normalized rating (0-100 scale) for easier interpretation
    movies['rating_100'] = movies['composite_rating'] * 10

    print(f"Computed composite ratings (0-10 scale): avg={movies['composite_rating'].mean():.2f}")
    print(f"Computed normalized ratings (0-100 scale): avg={movies['rating_100'].mean():.1f}")

    # Final column selection
    final_columns = [
        'movie_id', 'tconst', 'title', 'year', 'runtime',
        'genres', 'avg_rating', 'num_votes', 'composite_rating', 'rating_100', 'director', 'actors', 'country',
        'tmdb_id', 'tmdb_popularity', 'tmdb_rating', 'tmdb_vote_count',
        'description', 'poster_path', 'backdrop_path', 'poster_url', 'backdrop_url',
        'keywords', 'budget', 'revenue'
    ]
    movies = movies[final_columns]

    return movies


def main(imdb_output_dir: Path, processed_output_dir: Path):
    """
    Main execution: Transform IMDb data into internal app schema.

    Args:
        imdb_output_dir: Directory containing IMDb parquet files from data_clean.py
        processed_output_dir: Directory to save processed movies.parquet
    """
    print("Loading IMDb tables...")
    tables = load_imdb_tables(imdb_output_dir)

    print(f"Loaded {len(tables['movies_core'])} movies")
    print(f"Loaded {len(tables['movie_genres'])} genre entries")
    print(f"Loaded {len(tables['movie_ratings'])} ratings")
    print(f"Loaded {len(tables['movie_people_links'])} cast/crew links")
    print(f"Loaded {len(tables['people_core'])} people")

    print("\nBuilding internal movies table...")
    movies = build_internal_movies_table(tables)

    print(f"\nFinal movies table: {len(movies)} rows")
    print(f"Columns: {list(movies.columns)}")
    print(f"\nSample movie:")
    print(movies.iloc[0].to_dict())

    # Save processed table
    output_path = processed_output_dir / 'movies.parquet'
    movies.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return movies


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python transform_imdb_data.py <imdb_output_dir> <processed_output_dir>")
        print("Example: python transform_imdb_data.py ../../output ../../output/processed")
        sys.exit(1)

    imdb_dir = Path(sys.argv[1])
    processed_dir = Path(sys.argv[2])
    processed_dir.mkdir(parents=True, exist_ok=True)

    main(imdb_dir, processed_dir)
