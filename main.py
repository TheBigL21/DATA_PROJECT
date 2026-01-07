"""
Movie Recommendation System 

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
        import socket
        
        def find_available_port(start_port=5000, max_attempts=100):
            """Find the first available port starting from start_port"""
            for port in range(start_port, start_port + max_attempts):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', port))
                        return port
                except OSError:
                    continue
            raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")
        
        # Check if requested port is available, if not find another
        requested_port = args.port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', requested_port))
                actual_port = requested_port
                port_changed = False
        except OSError:
            print(f"⚠ Port {requested_port} is already in use.")
            actual_port = find_available_port(requested_port + 1)
            port_changed = True
            print(f"✓ Using alternative port: {actual_port}")
        
        # Write backend port to file for frontend to read
        backend_port_file = REPO_ROOT / '.backend-port'
        with open(backend_port_file, 'w') as f:
            f.write(str(actual_port))
        if port_changed:
            print(f"✓ Backend port ({actual_port}) written to .backend-port")
        
        print("=" * 60)
        print("STARTING API SERVER")
        print("=" * 60)
        print(f"Port: {actual_port}")
        print(f"Debug mode: {args.debug}")
        print("\nEndpoints available:")
        print(f"  - Health check: http://localhost:{actual_port}/health")
        print(f"  - API docs: See docs/API_DOCUMENTATION.md")
        print("\nThe frontend can now connect via Vite proxy.")
        if port_changed:
            print("⚠ NOTE: If frontend is already running, restart it to pick up the new port.")
        print("Press Ctrl+C to stop the server.\n")

        try:
            app.run(
                host='0.0.0.0',
                port=actual_port,
                debug=args.debug
            )
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            # Clean up port file on exit
            backend_port_file = REPO_ROOT / '.backend-port'
            if backend_port_file.exists():
                backend_port_file.unlink()
        except Exception as e:
            print(f"\nError starting server: {e}")
            print("\nMost common causes:")
            print("  1) Missing runtime artifacts (Quick Run):")
            print("     - You need the data package extracted into DATA_PROJECT/")
            print("     - Verify with: python -m src.verify_runtime_files")
            print("  2) You haven't generated artifacts yet (Full Pipeline):")
            print("     - Run: python main.py --pipeline  (takes hours)")


if __name__ == '__main__':
    main()
