"""
MAIN PIPELINE ORCHESTRATOR

This script runs the complete recommendation system pipeline from start to finish.

Pipeline Stages:
1. Data Cleaning (IMDb → parquet tables)
2. Data Transformation (parquet → internal app schema)
3. Synthetic Interaction Generation
4. Model Training:
   - Collaborative Filtering (ALS)
   - Co-occurrence Graph
5. System Test (generate sample recommendations)

Usage:
  python run_pipeline.py --imdb-dir /path/to/imdb/raw --output-dir /path/to/output

Options:
  --skip-cleaning: Skip data cleaning (assumes parquet files already exist)
  --skip-interactions: Skip interaction generation (use existing interactions)
  --num-users: Number of synthetic users to generate (default: 1000)
  --sessions-per-user: Sessions per user (default: 10)
  --latent-dim: CF model latent dimensions (default: 50)
  --iterations: CF model ALS iterations (default: 20)
"""

from pathlib import Path
import argparse
import sys
import subprocess
import os

# Compute repo root (this file is in src/, so go up one level)
REPO_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str):
    """
    Execute shell command and handle errors.

    Args:
        cmd: Command as list of strings
        description: Human-readable description for logging
    """
    print(f"\n{'='*60}")
    print(f"STAGE: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n{description} completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run complete recommendation system pipeline")

    # Required arguments
    parser.add_argument('--imdb-dir', type=str, required=True,
                        help='Directory containing IMDb .tsv.gz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for all generated files')

    # Optional flags
    parser.add_argument('--skip-cleaning', action='store_true',
                        help='Skip IMDb data cleaning step')
    parser.add_argument('--skip-tmdb', action='store_true',
                        help='Skip TMDb data enrichment step')
    parser.add_argument('--skip-interactions', action='store_true',
                        help='Skip interaction generation step')

    # Hyperparameters
    parser.add_argument('--num-users', type=int, default=1000,
                        help='Number of synthetic users to generate')
    parser.add_argument('--sessions-per-user', type=int, default=10,
                        help='Number of sessions per user')
    parser.add_argument('--latent-dim', type=int, default=50,
                        help='Latent dimensions for CF model')
    parser.add_argument('--iterations', type=int, default=20,
                        help='ALS iterations for CF model')

    args = parser.parse_args()

    # Setup paths
    imdb_dir = Path(args.imdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = output_dir / 'processed'
    models_dir = output_dir / 'models'

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    movies_parquet = processed_dir / 'movies.parquet'
    interactions_parquet = processed_dir / 'interactions.parquet'

    print("\n" + "="*60)
    print("MOVIE RECOMMENDATION SYSTEM - PIPELINE EXECUTION")
    print("="*60)
    print(f"IMDb directory: {imdb_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of users: {args.num_users}")
    print(f"Sessions per user: {args.sessions_per_user}")
    print(f"CF latent dimensions: {args.latent_dim}")
    print(f"CF iterations: {args.iterations}")

    # STAGE 1: Data Cleaning (IMDb TSV → Parquet)
    if not args.skip_cleaning:
        data_clean_path = REPO_ROOT / 'src' / 'data_processing' / 'data_clean.py'
        run_command(
            ['python3', str(data_clean_path), str(imdb_dir), str(output_dir)],
            "1. IMDb Data Cleaning"
        )
    else:
        print("\n[SKIPPING] Stage 1: IMDb Data Cleaning")

    # STAGE 1.5: Extract Country Data (from title.akas.tsv.gz)
    extract_country_path = REPO_ROOT / 'src' / 'data_processing' / 'extract_country_fast.py'
    run_command(
        ['python3', str(extract_country_path), str(imdb_dir), str(output_dir)],
        "1.5. Country Data Extraction"
    )

    # STAGE 1.6: Fetch TMDb Enrichment Data
    if not args.skip_tmdb:
        tmdb_api_key = os.getenv("TMDB_API_KEY", "a55b214aa396861a2625258556bbc6ee")
        fetch_tmdb_path = REPO_ROOT / 'src' / 'data_processing' / 'fetch_tmdb_data.py'
        run_command(
            ['python3', str(fetch_tmdb_path),
             str(output_dir / 'movies_core.parquet'), str(output_dir), tmdb_api_key],
            "1.6. TMDb Data Enrichment"
        )
    else:
        print("\n[SKIPPING] Stage 1.6: TMDb Data Enrichment")

    # STAGE 2: Data Transformation (Parquet → Internal Schema)
    transform_path = REPO_ROOT / 'src' / 'data_processing' / 'transform_imdb_data.py'
    run_command(
        ['python3', str(transform_path), str(output_dir), str(processed_dir)],
        "2. Data Transformation"
    )

    # STAGE 3: Synthetic Interaction Generation
    if not args.skip_interactions:
        generate_interactions_path = REPO_ROOT / 'src' / 'data_processing' / 'generate_synthetic_interactions.py'
        run_command(
            ['python3', str(generate_interactions_path),
             str(movies_parquet), str(processed_dir),
             str(args.num_users), str(args.sessions_per_user)],
            "3. Synthetic Interaction Generation"
        )
    else:
        print("\n[SKIPPING] Stage 3: Synthetic Interaction Generation")

    # STAGE 4: Train Collaborative Filtering Model
    cf_path = REPO_ROOT / 'src' / 'models' / 'collaborative_filtering.py'
    run_command(
        ['python3', str(cf_path),
         str(interactions_parquet), str(models_dir),
         str(args.latent_dim), str(args.iterations)],
        "4. Collaborative Filtering Model Training"
    )

    # STAGE 5: Build Co-occurrence Graph
    graph_path = REPO_ROOT / 'src' / 'models' / 'cooccurrence_graph.py'
    run_command(
        ['python3', str(graph_path),
         str(interactions_parquet), str(models_dir)],
        "5. Co-occurrence Graph Construction"
    )

    # STAGE 6: System Test
    print("\n" + "="*60)
    print("STAGE: 6. System Integration Test")
    print("="*60)

    # Import and test recommendation engine using package imports
    from src.recommendation.recommendation_engine import load_system

    try:
        engine = load_system(models_dir, movies_parquet)

        print("\n[TEST] Generating sample recommendations...")
        test_user_id = 0
        test_mood_genre = "action"
        test_mood_decade = "2000s"
        test_session_history = []

        recommendations = engine.recommend(
            test_user_id,
            test_mood_genre,
            test_mood_decade,
            test_session_history,
            top_k=5
        )

        print(f"\nSample recommendations for User {test_user_id}:")
        print(f"Mood: {test_mood_genre}, {test_mood_decade}")
        print("\nTop 5 movies:")
        for rank, movie_id in enumerate(recommendations, 1):
            movie = engine.movie_dict[movie_id]
            print(f"  {rank}. {movie['title']} ({movie['year']}) - {movie['genres'][:2]}")

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nGenerated files:")
        print(f"  - Movies: {movies_parquet}")
        print(f"  - Interactions: {interactions_parquet}")
        print(f"  - CF Model: {models_dir}/user_factors.npy, movie_factors.npy")
        print(f"  - Graph: {models_dir}/adjacency_matrix.npz")
        print(f"\nSystem is ready for deployment.")

    except Exception as e:
        print(f"\nERROR during system test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
