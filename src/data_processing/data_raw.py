# imdb_cleaning.py

import pandas as pd
from pathlib import Path

# -------- CONFIG --------
RAW_DIR = Path("/path/to/imdb/raw")      # folder where *.tsv.gz live
OUT_DIR = Path("/path/to/imdb/clean")    # folder where you want cleaned files
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_VOTES = 1000         # rating filter
MIN_RUNTIME = 30         # minutes
MAX_RUNTIME = 400
MOVIE_ONLY_TYPES = {"movie"}
KEEP_CATEGORIES = {"actor", "actress", "director", "writer"}
MAX_CAST = 5             # max actors per movie



### 1) IMPORT BASICS
basics_path = RAW_DIR / "title.basics.tsv.gz"
basics = pd.read_csv(
    basics_path,
    sep="\t",
    na_values="\\N",
    low_memory=False
)

# 1) CLEAN BASICS
# keep columns of interest
basics = basics[[
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "runtimeMinutes",
    "genres"
]]

# filter: movies only
basics = basics[basics["titleType"].isin(MOVIE_ONLY_TYPES)]

# non-adult only
basics = basics[basics["isAdult"] == 0]

# drop missing year, runtime, genres
basics = basics.dropna(subset=["startYear", "runtimeMinutes", "genres"])

# cast types
basics["startYear"] = basics["startYear"].astype(int)
basics["runtimeMinutes"] = basics["runtimeMinutes"].astype(int)

# filter unrealistic runtime
basics = basics[
    (basics["runtimeMinutes"] >= MIN_RUNTIME)
    & (basics["runtimeMinutes"] <= MAX_RUNTIME)
]

# normalize genres: lowercase, comma-separated, no '\N'
basics["genres"] = (
    basics["genres"]
    .str.replace(r"\\N", "", regex=True)
    .str.lower()
)

# drop rows where genres ended up empty
basics = basics[basics["genres"].str.len() > 0]

# drop columns no longer needed
basics = basics.drop(columns=["isAdult", "titleType"])

# save cleaned basics
basics.to_parquet(OUT_DIR / "movies_basics.parquet", index=False)

# 1b) DERIVE MOVIE-GENRES TABLE
tmp = basics[["tconst", "genres"]].copy()
tmp["genres"] = tmp["genres"].str.split(",")
movie_genres = tmp.explode("genres")
movie_genres = movie_genres.dropna(subset=["genres"])
movie_genres = movie_genres.drop_duplicates()

movie_genres.to_parquet(OUT_DIR / "movie_genres.parquet", index=False)



###2) IMPORT RATINGS
ratings_path = RAW_DIR / "title.ratings.tsv.gz"
ratings = pd.read_csv(
    ratings_path,
    sep="\t",
    na_values="\\N",
    low_memory=False
)

# 2) CLEAN RATINGS
# keep only movies that exist in cleaned basics
ratings = ratings[ratings["tconst"].isin(basics["tconst"])]

# filter by minimum vote count
ratings = ratings[ratings["numVotes"] >= MIN_VOTES]

# cast types
ratings["averageRating"] = ratings["averageRating"].astype(float)
ratings["numVotes"] = ratings["numVotes"].astype(int)

# save cleaned ratings
ratings.to_parquet(OUT_DIR / "movie_ratings.parquet", index=False)

# 2b) MERGE BASICS + RATINGS
movies = basics.merge(ratings, on="tconst", how="inner")

movies.to_parquet(OUT_DIR / "movies_core.parquet", index=False)



### 3) IMPORT PRINCIPALS
principals_path = RAW_DIR / "title.principals.tsv.gz"
principals = pd.read_csv(
    principals_path,
    sep="\t",
    na_values="\\N",
    low_memory=False
)

# 3) CLEAN PRINCIPALS
# retain only movies in movies_core
principals = principals[principals["tconst"].isin(movies["tconst"])]

# keep only relevant categories
principals = principals[principals["category"].isin(KEEP_CATEGORIES)]

# cast ordering to int
principals["ordering"] = principals["ordering"].astype(int)

# separate cast vs director/writer for more precise truncation
cast = principals[principals["category"].isin(["actor", "actress"])].copy()
crew = principals[principals["category"].isin(["director", "writer"])].copy()

# limit cast to top MAX_CAST actors/actresses per movie by ordering
cast = (
    cast.sort_values(["tconst", "ordering"])
        .groupby("tconst", as_index=False)
        .head(MAX_CAST)
)

# directors: usually 1, but keep all
directors = crew[crew["category"] == "director"].copy()

# writers: keep all
writers = crew[crew["category"] == "writer"].copy()

# optional: drop job and characters, keep only relationships
cols_keep = ["tconst", "nconst", "category", "ordering"]

cast = cast[cols_keep]
directors = directors[cols_keep]
writers = writers[cols_keep]

# re-concatenate
principals_clean = pd.concat([cast, directors, writers], ignore_index=True)

# save cleaned principals
principals_clean.to_parquet(OUT_DIR / "movie_people_links.parquet", index=False)



# 4) IMPORT NAMES
names_path = RAW_DIR / "name.basics.tsv.gz"
names = pd.read_csv(
    names_path,
    sep="\t",
    na_values="\\N",
    low_memory=False
)

# 4) CLEAN NAMES
# keep columns of interest
names = names[[
    "nconst",
    "primaryName",
    "birthYear"
]]

# keep only people referenced in principals_clean
used_nconst = principals_clean["nconst"].unique()
names = names[names["nconst"].isin(used_nconst)]

# normalize name string
names["primaryName"] = names["primaryName"].astype(str).str.strip()

# birthYear to numeric (nullable)
names["birthYear"] = pd.to_numeric(names["birthYear"], errors="coerce").astype("Int64")

# save cleaned names
names.to_parquet(OUT_DIR / "people_core.parquet", index=False)



#5) OPTIONAL CREW IMPORT, to add missing informations from other database (director,...)
crew_path = RAW_DIR / "title.crew.tsv.gz"
crew = pd.read_csv(
    crew_path,
    sep="\t",
    na_values="\\N",
    low_memory=False
)

# keep only movies in movies_core
crew = crew[crew["tconst"].isin(movies["tconst"])]

# split directors and writers into rows
def split_crew(df, col, category_label):
    tmp = df[["tconst", col]].dropna()
    tmp[col] = tmp[col].astype(str).str.split(",")
    tmp = tmp.explode(col)
    tmp = tmp.rename(columns={col: "nconst"})
    tmp["category"] = category_label
    return tmp[["tconst", "nconst", "category"]]

crew_directors = split_crew(crew, "directors", "director")
crew_writers = split_crew(crew, "writers", "writer")

crew_links = pd.concat([crew_directors, crew_writers], ignore_index=True).dropna()

# add an arbitrary ordering value for these fallback entries
crew_links["ordering"] = 9999

# determine which (movie, category) combos are missing in principals_clean
existing_keys = principals_clean[["tconst", "category"]].drop_duplicates()
crew_keys = crew_links[["tconst", "category"]].drop_duplicates()

missing_keys = (
    crew_keys
    .merge(existing_keys, on=["tconst", "category"], how="left", indicator=True)
)
missing_keys = missing_keys[missing_keys["_merge"] == "left_only"].drop(columns="_merge")

# keep only crew rows for missing (tconst, category) combos
crew_links = crew_links.merge(missing_keys, on=["tconst", "category"], how="inner")

# append to principals_clean
principals_augmented = pd.concat([principals_clean, crew_links], ignore_index=True)

principals_augmented.to_parquet(OUT_DIR / "movie_people_links_augmented.parquet", index=False)
