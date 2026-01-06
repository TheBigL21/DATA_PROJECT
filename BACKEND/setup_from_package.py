"""
Setup Movie Finder from a data package zip (Quick Run / no pipeline).

This script extracts the zip into DATA_PROJECT/ and then runs verification.

Usage:
  python setup_from_package.py --package /path/to/movie_finder_data_package_*.zip
  python setup_from_package.py --package ./movie_finder_data_package_*.zip --force
"""

from __future__ import annotations

from pathlib import Path
import argparse
import zipfile
import sys


def _is_safe_member(name: str) -> bool:
    # Reject absolute paths and path traversal
    if name.startswith("/") or name.startswith("\\"):
        return False
    parts = Path(name).parts
    return ".." not in parts


def main() -> int:
    project_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Extract Movie Finder data package into DATA_PROJECT/")
    parser.add_argument("--package", required=True, help="Path to the data package zip")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if present (recommended if re-extracting).",
    )
    args = parser.parse_args()

    package_path = Path(args.package).expanduser()
    if not package_path.exists():
        print(f"ERROR: Package not found: {package_path.as_posix()}")
        return 1

    print("=" * 60)
    print("MOVIE FINDER - SETUP FROM DATA PACKAGE")
    print("=" * 60)
    print(f"Package:     {package_path.as_posix()}")
    print(f"Destination: {project_dir.as_posix()}\n")

    with zipfile.ZipFile(package_path, "r") as zf:
        members = zf.namelist()
        unsafe = [m for m in members if not _is_safe_member(m)]
        if unsafe:
            print("ERROR: Zip contains unsafe paths (refusing to extract):")
            for m in unsafe[:20]:
                print(f"  - {m}")
            return 1

        # Extract file-by-file to support --force consistently
        for member in members:
            if member.endswith("/"):
                continue

            dest_path = project_dir / member
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists() and not args.force:
                # Skip existing files unless forced
                continue

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

    print("Extraction complete.\n")

    # Run verification
    try:
        from verify_runtime_files import main as verify_main

        print("Running verification...\n")
        return int(verify_main())
    except Exception as e:
        print(f"WARNING: Could not run verification script automatically: {e}")
        print("You can run it manually with: python verify_runtime_files.py")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


