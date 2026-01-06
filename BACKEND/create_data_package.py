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
    project_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Create Movie Finder data package zip (runtime artifacts only)")
    parser.add_argument(
        "--out",
        default=f"movie_finder_data_package_{time.strftime('%Y%m%d')}.zip",
        help="Output zip filename (created in DATA_PROJECT/ unless an absolute path is provided)",
    )
    parser.add_argument(
        "--include-feedback-db",
        action="store_true",
        help="Include output/feedback.db if it exists (otherwise it is omitted; it will be created at runtime).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_dir / out_path

    required = {
        "config/genre_allocation.json": project_dir / "config" / "genre_allocation.json",
        "output/processed/movies.parquet": project_dir / "output" / "processed" / "movies.parquet",
        "output/models/user_factors.npy": project_dir / "output" / "models" / "user_factors.npy",
        "output/models/movie_factors.npy": project_dir / "output" / "models" / "movie_factors.npy",
        "output/models/model_metadata.json": project_dir / "output" / "models" / "model_metadata.json",
        "output/models/adjacency_matrix.npz": project_dir / "output" / "models" / "adjacency_matrix.npz",
        "output/models/graph_metadata.json": project_dir / "output" / "models" / "graph_metadata.json",
    }

    optional = {
        "output/models/keyword_database.pkl": project_dir / "output" / "models" / "keyword_database.pkl",
        "output/models/content_similarity.pkl": project_dir / "output" / "models" / "content_similarity.pkl",
    }

    missing_required = [k for k, v in required.items() if not v.exists()]
    if missing_required:
        print("ERROR: Cannot create package; missing required artifacts:\n")
        for k in missing_required:
            print(f"  - {k}")
        print("\nFix:")
        print("  - Extract a data package first, OR run the pipeline to generate artifacts.")
        print("  - You can verify your runtime files with: python verify_runtime_files.py")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": "movie_finder_data_package",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "includes_optional": [],
        "notes": "Extract into DATA_PROJECT/ to run backend without pipeline.",
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
        feedback_db = project_dir / "output" / "feedback.db"
        if args.include_feedback_db and feedback_db.exists():
            zf.write(feedback_db, arcname="output/feedback.db")
            manifest["includes_optional"].append("output/feedback.db")

        # Add manifest
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    print("Done.\n")
    print("Next steps for the recipient (TA):")
    print("  1) unzip the file into DATA_PROJECT/")
    print("  2) python verify_runtime_files.py")
    print("  3) python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


