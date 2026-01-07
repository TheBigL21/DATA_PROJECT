import sys
from pathlib import Path
from typing import List

import pandas as pd


def read_tsv_gz(path: Path, usecols: List[str]) -> pd.DataFrame:
    """
    Read a gzipped TSV file with all columns as strings, then subset.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype=str,
        na_filter=False,  # keep '\N' as literal string
    )
    return df[usecols]


def clean_title_basics(basics_path: Path) -> pd.DataFrame:
    """
    Load and clean title.basics according to the specification.
    Returns only movie rows that pass:
      - titleType == "movie"
      - isAdult == "0"
      - 30 <= runtimeMinutes <= 400
      - startYear != '\\N'
      - genres != '\\N'
    """
    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres",
    ]
    df = read_tsv_gz(basics_path, usecols)

    # Filter to movies and non-adult
    df = df[(df["titleType"] == "movie") & (df["isAdult"] == "0")]

    # Filter startYear not missing
    df = df[df["startYear"] != r"\N"]

    # Convert startYear to int
    df["startYear"] = df["startYear"].astype(int)

    # Filter runtimeMinutes not missing and within bounds
    df = df[df["runtimeMinutes"] != r"\N"]
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df = df[df["runtimeMinutes"].between(30, 400, inclusive="both")]

    # Filter genres not missing
    df = df[df["genres"] != r"\N"]

    # Normalize genres to lowercase, but we keep them as a comma-separated string for now
    df["genres"] = df["genres"].str.lower()

    # Keep only necessary columns for movies_core later
    df = df[["tconst", "primaryTitle", "originalTitle", "startYear", "runtimeMinutes", "genres"]]

    df = df.drop_duplicates(subset="tconst")

    return df


def clean_title_ratings(ratings_path: Path, valid_tconsts: pd.Series) -> pd.DataFrame:
    """
    Load title.ratings, filter by numVotes >= 1000, and restrict to movies in valid_tconsts.
    """
    usecols = ["tconst", "averageRating", "numVotes"]
    df = read_tsv_gz(ratings_path, usecols)

    # Filter numVotes >= 1000 and not missing
    df = df[df["numVotes"] != r"\N"]
    df["numVotes"] = df["numVotes"].astype(int)
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")

    df = df[df["numVotes"] >= 1000]

    # Restrict to movies coming from basics
    df = df[df["tconst"].isin(valid_tconsts)]

    # Drop rows with missing rating
    df = df.dropna(subset=["averageRating"])

    df = df.drop_duplicates(subset="tconst")

    return df


def clean_title_principals(principals_path: Path, valid_tconsts: pd.Series) -> pd.DataFrame:
    """
    Load title.principals, keep only desired categories (actor, actress, director, writer)
    and movies in valid_tconsts. Limit actors/actresses to top 5 per movie by ordering.
    """
    usecols = ["tconst", "nconst", "category", "ordering"]
    df = read_tsv_gz(principals_path, usecols)

    # Filter by categories
    allowed_categories = {"actor", "actress", "director", "writer"}
    df = df[df["category"].isin(allowed_categories)]

    # Restrict to tconsts present in current movie set
    df = df[df["tconst"].isin(valid_tconsts)]

    # Convert ordering to int (missing ordering should not occur in principals)
    df = df[df["ordering"] != r"\N"]
    df["ordering"] = df["ordering"].astype(int)

    # Split into actors (actor+actress) and other categories
    is_actor = df["category"].isin(["actor", "actress"])
    actors = df[is_actor]
    others = df[~is_actor]

    # For actors/actresses: top 5 per movie by ordering
    actors = (
        actors.sort_values(["tconst", "ordering"])
        .groupby("tconst", as_index=False)
        .head(5)
    )

    principals_clean = pd.concat([actors, others], ignore_index=True)
    principals_clean = principals_clean.drop_duplicates(subset=["tconst", "nconst", "category", "ordering"])

    return principals_clean


def patch_with_crew(
    crew_path: Path,
    principals_clean: pd.DataFrame,
    movie_tconsts: pd.Series,
) -> pd.DataFrame:
    """
    Use title.crew to patch missing directors (and writers) for movies in movie_tconsts.
    - For each movie lacking a director:
        add rows from crew.directors as category="director", ordering=0
    - For each movie lacking a writer:
        add rows from crew.writers as category="writer", ordering=0
    Returns the union of principals_clean and patch rows.
    """
    usecols = ["tconst", "directors", "writers"]
    crew_df = read_tsv_gz(crew_path, usecols)
    crew_df = crew_df[crew_df["tconst"].isin(movie_tconsts)]

    principals = principals_clean.copy()

    # Helper: find which movies already have directors/writers
    has_director = principals[principals["category"] == "director"]["tconst"].unique()
    has_writer = principals[principals["category"] == "writer"]["tconst"].unique()

    has_director_set = set(has_director)
    has_writer_set = set(has_writer)

    patch_rows = []

    for _, row in crew_df.iterrows():
        tconst = row["tconst"]

        # Patch directors if none present
        if tconst not in has_director_set:
            directors = row["directors"]
            if directors and directors != r"\N":
                for n in directors.split(","):
                    patch_rows.append(
                        {
                            "tconst": tconst,
                            "nconst": n,
                            "category": "director",
                            "ordering": 0,
                        }
                    )

        # Patch writers if none present
        if tconst not in has_writer_set:
            writers = row["writers"]
            if writers and writers != r"\N":
                for n in writers.split(","):
                    patch_rows.append(
                        {
                            "tconst": tconst,
                            "nconst": n,
                            "category": "writer",
                            "ordering": 0,
                        }
                    )

    if patch_rows:
        patch_df = pd.DataFrame(patch_rows)
        # Ensure correct dtypes
        patch_df["ordering"] = patch_df["ordering"].astype(int)
        principals = pd.concat([principals, patch_df], ignore_index=True)

    principals = principals.drop_duplicates(subset=["tconst", "nconst", "category", "ordering"])

    return principals


def enforce_required_cast(principals_patched: pd.DataFrame, movie_tconsts: pd.Series) -> pd.Series:
    """
    Given patched principals and candidate movie_tconsts, keep only movies with:
      - at least one actor/actress
      - at least one director
    Returns the filtered set of tconsts.
    """
    df = principals_patched.copy()
    df = df[df["tconst"].isin(movie_tconsts)]

    # Compute flags per movie
    has_actor = (
        df[df["category"].isin(["actor", "actress"])]
        .groupby("tconst")
        .size()
        .rename("actor_count")
    )
    has_director = (
        df[df["category"] == "director"]
        .groupby("tconst")
        .size()
        .rename("director_count")
    )

    counts = pd.concat([has_actor, has_director], axis=1).fillna(0)
    counts["actor_count"] = counts["actor_count"].astype(int)
    counts["director_count"] = counts["director_count"].astype(int)

    valid = counts[(counts["actor_count"] > 0) & (counts["director_count"] > 0)]
    valid_tconsts = valid.index.to_series()

    return valid_tconsts


def build_movie_genres(basics_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build movie_genres table: one row per (tconst, genre).
    Assumes basics_clean['genres'] is a comma-separated string of lowercase genres.
    """
    rows = []
    for _, row in basics_clean.iterrows():
        tconst = row["tconst"]
        genres_str = row["genres"]
        for g in genres_str.split(","):
            g = g.strip()
            if g:
                rows.append({"tconst": tconst, "genre": g})

    movie_genres = pd.DataFrame(rows)
    movie_genres = movie_genres.drop_duplicates()

    return movie_genres


def clean_name_basics(names_path: Path, used_nconsts: pd.Series) -> pd.DataFrame:
    """
    Load name.basics, keep only nconst present in used_nconsts.
    Keep columns: nconst, primaryName, birthYear.
    Normalize primaryName.
    """
    usecols = ["nconst", "primaryName", "birthYear"]
    df = read_tsv_gz(names_path, usecols)

    df = df[df["nconst"].isin(used_nconsts)]

    # Normalize names
    df["primaryName"] = df["primaryName"].str.strip()

    # Handle birthYear
    df["birthYear"] = df["birthYear"].replace(r"\N", pd.NA)
    # birthYear is optional, keep as nullable string or convert to Int64 if you prefer
    # df["birthYear"] = pd.to_numeric(df["birthYear"], errors="coerce").astype("Int64")

    df = df.drop_duplicates(subset="nconst")

    return df


def main(imdb_dir: Path, output_dir: Path) -> None:
    imdb_dir = imdb_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    basics_path = imdb_dir / "title.basics.tsv.gz"
    ratings_path = imdb_dir / "title.ratings.tsv.gz"
    principals_path = imdb_dir / "title.principals.tsv.gz"
    crew_path = imdb_dir / "title.crew.tsv.gz"
    names_path = imdb_dir / "name.basics.tsv.gz"

    # 1. Clean basics
    basics_clean = clean_title_basics(basics_path)
    print(f"[basics] movies after filtering: {len(basics_clean)}")

    # 2. Clean ratings and join
    ratings_clean = clean_title_ratings(ratings_path, basics_clean["tconst"])
    print(f"[ratings] movies with sufficient votes: {len(ratings_clean)}")

    # Inner join to ensure required rating fields present
    movies = basics_clean.merge(ratings_clean, on="tconst", how="inner")
    print(f"[movies] after join basics+ratings: {len(movies)}")

    # 3. Clean principals
    principals_clean = clean_title_principals(principals_path, movies["tconst"])
    print(f"[principals] rows after cleaning: {len(principals_clean)}")

    # 4. Patch with crew
    principals_patched = patch_with_crew(crew_path, principals_clean, movies["tconst"])
    print(f"[principals] rows after patch: {len(principals_patched)}")

    # 5. Enforce required cast/crew (at least one actor + one director)
    valid_tconsts = enforce_required_cast(principals_patched, movies["tconst"])
    print(f"[movies] with required actor+director: {len(valid_tconsts)}")

    # Restrict movies, ratings, principals to valid_tconsts
    movies = movies[movies["tconst"].isin(valid_tconsts)].copy()
    ratings_final = ratings_clean[ratings_clean["tconst"].isin(valid_tconsts)].copy()
    principals_final = principals_patched[principals_patched["tconst"].isin(valid_tconsts)].copy()

    # 6. Build movies_core
    movies_core = movies[["tconst", "primaryTitle", "originalTitle", "startYear", "runtimeMinutes"]].drop_duplicates(
        subset="tconst"
    )

    # 7. Build movie_genres
    movie_genres = build_movie_genres(movies[["tconst", "genres"]].drop_duplicates(subset="tconst"))

    # 8. movie_ratings
    movie_ratings = ratings_final[["tconst", "averageRating", "numVotes"]].drop_duplicates(subset="tconst")

    # 9. movie_people_links (actors + directors + writers, fully patched)
    movie_people_links = principals_final[["tconst", "nconst", "category", "ordering"]].copy()
    movie_people_links = movie_people_links.drop_duplicates()

    # 10. people_core (from name.basics)
    used_nconsts = movie_people_links["nconst"].unique()
    people_core = clean_name_basics(names_path, pd.Series(used_nconsts))

    # === Assertions: required fields completeness ===

    # startYear, runtimeMinutes must not be missing
    assert movies_core["startYear"].notna().all(), "Missing startYear in movies_core"
    assert movies_core["runtimeMinutes"].notna().all(), "Missing runtimeMinutes in movies_core"

    # ratings required
    assert movie_ratings["averageRating"].notna().all(), "Missing averageRating in movie_ratings"
    assert movie_ratings["numVotes"].notna().all(), "Missing numVotes in movie_ratings"

    # at least one genre per movie: verify every tconst in movies_core appears in movie_genres
    genre_counts = movie_genres.groupby("tconst").size()
    assert set(movies_core["tconst"]).issubset(set(genre_counts[genre_counts > 0].index)), \
        "Some movies have no genres in movie_genres"

    # at least one actor and at least one director per movie
    actor_counts = (
        movie_people_links[movie_people_links["category"].isin(["actor", "actress"])]
        .groupby("tconst")
        .size()
    )
    director_counts = (
        movie_people_links[movie_people_links["category"] == "director"]
        .groupby("tconst")
        .size()
    )

    missing_actor = set(movies_core["tconst"]) - set(actor_counts[actor_counts > 0].index)
    missing_director = set(movies_core["tconst"]) - set(director_counts[director_counts > 0].index)

    assert not missing_actor, f"Some movies are missing actors: {len(missing_actor)}"
    assert not missing_director, f"Some movies are missing directors: {len(missing_director)}"

    # === Write parquet outputs ===
    movies_core.to_parquet(output_dir / "movies_core.parquet", index=False)
    movie_genres.to_parquet(output_dir / "movie_genres.parquet", index=False)
    movie_ratings.to_parquet(output_dir / "movie_ratings.parquet", index=False)
    movie_people_links.to_parquet(output_dir / "movie_people_links.parquet", index=False)
    people_core.to_parquet(output_dir / "people_core.parquet", index=False)

    print(f"Written parquet tables to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_imdb_tables.py /path/to/imdb_dumps /path/to/output_dir")
        sys.exit(1)

    imdb_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    main(imdb_dir, output_dir)
