"""
Verify Movie Finder runtime artifacts exist (Quick Run / no pipeline).

This script checks that the required files from the "data package" are present
inside DATA_PROJECT/, and that they are readable.

Usage:
  python verify_runtime_files.py

Exit codes:
  0 = OK
  1 = Missing required files or unreadable artifacts
"""

from __future__ import annotations

from pathlib import Path
import sys
import json


def _human_list(paths: list[Path]) -> str:
    return "\n".join(f"  - {p.as_posix()}" for p in paths)


def main() -> int:
    # Compute repo root (this file is in src/, so go up one level)
    REPO_ROOT = Path(__file__).parent.parent

    required_files = [
        REPO_ROOT / "src" / "config" / "genre_allocation.json",
        REPO_ROOT / "data" / "processed" / "movies.parquet",
        REPO_ROOT / "data" / "models" / "user_factors.npy",
        REPO_ROOT / "data" / "models" / "movie_factors.npy",
        REPO_ROOT / "data" / "models" / "model_metadata.json",
        REPO_ROOT / "data" / "models" / "adjacency_matrix.npz",
        REPO_ROOT / "data" / "models" / "graph_metadata.json",
    ]

    optional_files = [
        REPO_ROOT / "data" / "models" / "keyword_database.pkl",
        REPO_ROOT / "data" / "models" / "content_similarity.pkl",
        REPO_ROOT / "data" / "feedback.db",
    ]

    missing_required = [p for p in required_files if not p.exists()]

    print("=" * 60)
    print("MOVIE FINDER - RUNTIME FILE VERIFICATION")
    print("=" * 60)
    print(f"Project dir: {REPO_ROOT.as_posix()}\n")

    if missing_required:
        print("ERROR: Missing required runtime files:\n")
        print(_human_list(missing_required))
        print("\nHow to fix (Quick Run):")
        print("  1) Download the data package zip (movie_finder_data_package_*.zip)")
        print("  2) Extract it into DATA_PROJECT/ so it creates:")
        print("     - DATA_PROJECT/data/... and DATA_PROJECT/src/config/genre_allocation.json")
        print("\nAlternative fix (slow):")
        print("  - Run the full pipeline: python main.py --pipeline")
        return 1

    # Lightweight readability checks (actionable failures)
    try:
        with open(REPO_ROOT / "src" / "config" / "genre_allocation.json", "r") as f:
            json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read/parse src/config/genre_allocation.json: {e}")
        return 1

    # Check parquet can be read (requires pyarrow/fastparquet already in requirements)
    try:
        import pandas as pd  # type: ignore

        movies_path = REPO_ROOT / "data" / "processed" / "movies.parquet"
        df = pd.read_parquet(movies_path)
        # Minimal schema sanity checks used by the API
        for col in ["movie_id", "title", "year", "genres", "avg_rating", "num_votes"]:
            if col not in df.columns:
                raise ValueError(f"movies.parquet missing expected column '{col}'")
        print(f"OK: movies.parquet loaded ({len(df):,} rows)")
    except Exception as e:
        print(f"ERROR: Could not read data/processed/movies.parquet: {e}")
        print("Tip: this usually means the data package is incomplete/corrupted, or parquet deps are missing.")
        return 1

    # Optional files: report status only
    present_optional = [p for p in optional_files if p.exists()]
    missing_optional = [p for p in optional_files if not p.exists()]

    print("\nRequired files: OK")
    if present_optional:
        print("\nOptional files found:")
        print(_human_list(present_optional))
    if missing_optional:
        print("\nOptional files missing (this is fine):")
        print(_human_list(missing_optional))

    print("\nNext steps:")
    print("  - Start backend:  python main.py")
    print("  - Health check:   http://localhost:5000/health")
    print("  - Start frontend: npm --prefix frontend run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


