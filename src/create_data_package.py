"""
Create a minimal "data package" zip for Quick Run (no pipeline).

This zip is meant to be distributed outside git (e.g., GitHub Release asset).
It contains only the runtime artifacts required by the backend.

Usage:
  python create_data_package.py
  python create_data_package.py --out movie_finder_data_package_v1.zip
  python create_data_package.py --out movie_finder_data_package_v1.zip --include-feedback-db
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time
import zipfile


def _add_file(zf: zipfile.ZipFile, src: Path, arcname: str):
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    zf.write(src, arcname=arcname)


def main() -> int:
    # Compute repo root (this file is in src/, so go up one level)
    REPO_ROOT = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Create Movie Finder data package zip (runtime artifacts only)")
    parser.add_argument(
        "--out",
        default=f"movie_finder_data_package_{time.strftime('%Y%m%d')}.zip",
        help="Output zip filename (created in repo root unless an absolute path is provided)",
    )
    parser.add_argument(
        "--include-feedback-db",
        action="store_true",
        help="Include results/feedback.db if it exists (otherwise it is omitted; it will be created at runtime).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    required = {
        "src/config/genre_allocation.json": REPO_ROOT / "src" / "config" / "genre_allocation.json",
        "results/processed/movies.parquet": REPO_ROOT / "results" / "processed" / "movies.parquet",
        "results/models/user_factors.npy": REPO_ROOT / "results" / "models" / "user_factors.npy",
        "results/models/movie_factors.npy": REPO_ROOT / "results" / "models" / "movie_factors.npy",
        "results/models/model_metadata.json": REPO_ROOT / "results" / "models" / "model_metadata.json",
        "results/models/adjacency_matrix.npz": REPO_ROOT / "results" / "models" / "adjacency_matrix.npz",
        "results/models/graph_metadata.json": REPO_ROOT / "results" / "models" / "graph_metadata.json",
    }

    optional = {
        "results/models/keyword_database.pkl": REPO_ROOT / "results" / "models" / "keyword_database.pkl",
        "results/models/content_similarity.pkl": REPO_ROOT / "results" / "models" / "content_similarity.pkl",
    }

    missing_required = [k for k, v in required.items() if not v.exists()]
    if missing_required:
        print("ERROR: Cannot create package; missing required artifacts:\n")
        for k in missing_required:
            print(f"  - {k}")
        print("\nFix:")
        print("  - Extract a data package first, OR run the pipeline to generate artifacts.")
        print("  - You can verify your runtime files with: python -m src.verify_runtime_files")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": "movie_finder_data_package",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "includes_optional": [],
        "notes": "Extract into repo root to run backend without pipeline.",
    }

    print(f"Creating data package: {out_path.as_posix()}")
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Required files
        for arc, src in required.items():
            _add_file(zf, src, arcname=arc)

        # Optional files (include if present)
        for arc, src in optional.items():
            if src.exists():
                zf.write(src, arcname=arc)
                manifest["includes_optional"].append(arc)

        # Optional feedback DB (only if flag enabled and file exists)
        feedback_db = REPO_ROOT / "results" / "feedback.db"
        if args.include_feedback_db and feedback_db.exists():
            zf.write(feedback_db, arcname="results/feedback.db")
            manifest["includes_optional"].append("results/feedback.db")

        # Add manifest
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print("Done.\n")
    print("Next steps for the recipient (TA):")
    print("  1) unzip the file into repo root")
    print("  2) python -m src.verify_runtime_files")
    print("  3) python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


