"""
Movie Recommendation System - Main Entry Point


Usage:
    python main.py                    # Start API server (for frontend)
    python main.py --cli              # Run simple CLI demo
    python main.py --interactive      # Run interactive swipe CLI
    python main.py --pipeline         # Run data pipeline (2-4 hours)
    python main.py --help             # Show all options
"""

import sys
import argparse
from pathlib import Path

# Compute repo root
REPO_ROOT = Path(__file__).parent

# Prefer package imports from src/
from src.api_smart import app
from src.run_pipeline import main as run_pipeline_main
from src.movie_finder import main as movie_finder_main
from src.interactive_movie_finder import main as interactive_movie_finder_main


def main():
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System - Hybrid ML approach for personalized recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start API server on port 5000
  python main.py --cli              # Run CLI questionnaire (simple)
  python main.py --interactive      # Run swipe interface (advanced)
  python main.py --pipeline         # Build dataset from scratch
  python main.py --port 8000        # Start API on custom port

The API server is needed for the frontend to work.
CLI tools work without any frontend setup.
        """
    )

    parser.add_argument('--cli', action='store_true',
                       help='Run simple CLI questionnaire (no frontend needed)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive swipe CLI (no frontend needed)')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run full data pipeline (WARNING: takes 2-4 hours)')
    parser.add_argument('--port', type=int, default=5000,
                       help='API server port (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Run API server in debug mode')

    args = parser.parse_args()

    # Run data pipeline
    if args.pipeline:
        print("=" * 60)
        print("RUNNING DATA PIPELINE")
        print("=" * 60)
        print("This will:")
        print("  1. Download IMDb datasets (~10GB)")
        print("  2. Clean and filter data")
        print("  3. Fetch TMDb metadata (2-4 hours)")
        print("  4. Train ML models")
        print("\nThis may take 2-4 hours. Continue? (y/n)")

        response = input().strip().lower()
        if response != 'y':
            print("Pipeline cancelled.")
            return

        print("\nStarting pipeline...")
        run_pipeline_main()
        print("\nPipeline complete!")

    # Run simple CLI
    elif args.cli:
        print("=" * 60)
        print("SIMPLE CLI QUESTIONNAIRE")
        print("=" * 60)
        print("Answer a few questions to get 10 movie recommendations.\n")

        movie_finder_main()

    # Run interactive CLI
    elif args.interactive:
        print("=" * 60)
        print("INTERACTIVE SWIPE INTERFACE")
        print("=" * 60)
        print("Swipe through movies like Tinder!")
        print("  ← (left)  = No thanks")
        print("  → (right) = Maybe")
        print("  ↑ (up)    = Perfect!\n")

        interactive_movie_finder_main()

    # Default: Start API server
    else:
        print("=" * 60)
        print("STARTING API SERVER")
        print("=" * 60)
        print(f"Port: {args.port}")
        print(f"Debug mode: {args.debug}")
        print("\nEndpoints available:")
        print(f"  - Health check: http://localhost:{args.port}/health")
        print(f"  - API docs: See docs/API_DOCUMENTATION.md")
        print("\nThe frontend can now connect to this server.")
        print("Press Ctrl+C to stop the server.\n")

        try:
            app.run(
                host='0.0.0.0',
                port=args.port,
                debug=args.debug
            )
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
        except Exception as e:
            print(f"\nError starting server: {e}")
            print("\nMost common causes:")
            print("  1) Missing runtime artifacts (Quick Run):")
            print("     - You need the data package extracted into DATA_PROJECT/")
            print("     - Verify with: python -m src.verify_runtime_files or python src/verify_runtime_files.py")
            print("  2) You haven't generated artifacts yet (Full Pipeline):")
            print("     - Run: python main.py --pipeline  (takes hours)")
            print("  3) Port already in use:")
            print(f"     - Try: python main.py --port 8000  (and set VITE_API_URL accordingly)")


if __name__ == '__main__':
    main()
