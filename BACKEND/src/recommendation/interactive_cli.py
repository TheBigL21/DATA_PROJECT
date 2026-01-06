"""
INTERACTIVE MOVIE SELECTOR - CLI INTERFACE

Command-line interface for interactive movie selection with learning.

Usage:
    python -m src.recommendation.interactive_cli
"""

from pathlib import Path
import sys
import uuid
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from recommendation.smart_engine import load_smart_system
from recommendation.interactive_selector import (
    FeatureEncoder,
    SessionLearner,
    SequenceLearner,
    GlobalLearner,
    AdaptivePoolManager,
    InteractiveSelector,
    LearningStorage,
    FeedbackEvent,
    display_movie_card,
    display_movie_details,
    action_to_satisfaction
)


class InteractiveSession:
    """Manage an interactive movie selection session"""

    def __init__(
        self,
        smart_engine,
        encoder: FeatureEncoder,
        selector: InteractiveSelector,
        storage: LearningStorage,
        user_id: int = 0
    ):
        self.smart_engine = smart_engine
        self.encoder = encoder
        self.selector = selector
        self.storage = storage
        self.user_id = user_id

        self.session_id = str(uuid.uuid4())[:8]
        self.context = {}
        self.pool_manager = None
        self.session_history = []

    def collect_context(self) -> dict:
        """Collect user context (4 questions)"""
        print("\n" + "="*60)
        print("MOVIE RECOMMENDATION - INTERACTIVE MODE")
        print("="*60 + "\n")

        # Question 1: Genres
        print("1. What genres are you interested in? (select 1-2)")
        print("   Options: action, thriller, comedy, drama, horror, sci-fi, romance, adventure")
        genres_input = input("   Enter genres (comma-separated): ").strip().lower()
        genres = [g.strip() for g in genres_input.split(',')][:2]

        if not genres:
            genres = ['action']  # Default

        # Question 2: Themes (optional keywords)
        print("\n2. Any specific themes? (optional, comma-separated)")
        print("   Examples: heist, espionage, revenge, family, coming-of-age")
        themes_input = input("   Enter themes: ").strip().lower()
        themes = [t.strip() for t in themes_input.split(',') if t.strip()] if themes_input else []

        # Question 3: Era
        print("\n3. What time period?")
        print("   Options:")
        print("     new_gen (2010+) - Fresh, modern movies")
        print("     golden_era (2000-2009) - Classic 2000s cinema")
        print("     throwback (1990s) - Iconic 90s films")
        print("     old_school (before 1990) - Vintage classics")
        print("     any - No preference")
        era = input("   Enter era: ").strip().lower()

        if era not in ['new_gen', 'golden_era', 'throwback', 'old_school', 'any']:
            era = 'golden_era'  # Default

        # Question 4: Source material
        print("\n4. Source material preference?")
        print("   Options: book, any")
        source_material = input("   Enter preference: ").strip().lower()

        if source_material not in ['book']:
            source_material = 'any'

        context = {
            'genres': genres,
            'era': era,
            'source_material': source_material,
            'themes': themes
        }

        print(f"\n✓ Context collected: {context}\n")
        return context

    def initialize_pool(self, context: dict) -> AdaptivePoolManager:
        """Generate initial candidate pool using SmartEngine with strict filtering"""
        # Get initial candidates from SmartEngine (with hard filters)
        initial_candidates = self.smart_engine.recommend(
            user_id=self.user_id,
            genres=context['genres'],
            era=context['era'],
            source_material=context.get('source_material', 'any'),
            themes=context['themes'],
            session_history=[],
            top_k=100  # Get 100 initial candidates
        )

        # Get strict scores from SmartEngine (for 80/20 weighting)
        strict_scores = self.smart_engine.get_strict_scores()

        # Pass strict scores to InteractiveSelector
        self.selector.set_strict_scores(strict_scores)

        # Create adaptive pool manager
        pool_manager = AdaptivePoolManager(
            encoder=self.encoder,
            movies_df=self.smart_engine.movies,
            initial_candidates=initial_candidates,
            context=context,
            pool_size_target=80
        )

        print(f"✓ Generated strict pool: {len(initial_candidates)} candidates")
        print(f"✓ Quality floor: 6.0, Votes floor: 5000")
        print(f"✓ Scoring: 80% strict + 20% adaptive\n")

        return pool_manager

    def run_session(self):
        """Run interactive selection loop"""
        # Step 1: Collect context
        self.context = self.collect_context()

        # Step 2: Initialize pool
        self.pool_manager = self.initialize_pool(self.context)

        # Step 3: Interactive loop
        position = 1
        last_movie_id = None

        print("="*60)
        print("STARTING INTERACTIVE SELECTION")
        print("="*60 + "\n")

        while True:
            # Select next movie
            movie_id = self.selector.select_next(
                pool_manager=self.pool_manager,
                context=self.context,
                last_movie_id=last_movie_id
            )

            if movie_id is None:
                print("\n⚠ No more candidates available. Ending session.\n")
                break

            # Display movie
            movie = self.smart_engine.movies[
                self.smart_engine.movies['movie_id'] == movie_id
            ].iloc[0]

            display_movie_card(movie, position)

            # Get user input
            user_input = input("Choice: ").strip()

            # Map input to action
            if user_input == '' or user_input == '0':
                action = 'yes'
                print("→ Yes, show similar movies\n")
            elif user_input == '1':
                action = 'no'
                print("→ No, show different movies\n")
            elif user_input == '2':
                action = 'final'
                print("→ FINAL CHOICE!\n")
            else:
                print("⚠ Invalid input, treating as 'yes'\n")
                action = 'yes'

            # Mark as shown
            self.pool_manager.mark_shown(movie_id)

            # Process feedback
            self.selector.process_feedback(
                movie_id=movie_id,
                action=action,
                context=self.context,
                pool_manager=self.pool_manager,
                last_movie_id=last_movie_id
            )

            # Log event
            event = FeedbackEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                context=self.context,
                movie_id=movie_id,
                action=action,
                satisfaction=action_to_satisfaction(action),
                position_in_session=position,
                previous_movie_id=last_movie_id
            )
            self.storage.log_event(event)

            # Update state
            last_movie_id = movie_id
            position += 1

            # If final choice, display details and end
            if action == 'final':
                display_movie_details(movie)
                break

            # Safety: max 20 movies per session
            if position > 20:
                print("\n⚠ Reached maximum session length. Ending.\n")
                break

        # Session summary
        self.print_session_summary()

    def print_session_summary(self):
        """Print session summary and learning insights"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)

        total_shown = len(self.session_history)
        print(f"Movies shown: {total_shown}")

        # Get learned feature preferences
        top_features = self.selector.session_learner.get_top_features(top_k=5)

        print("\nLearned preferences:")
        for feature_name, weight in top_features:
            sentiment = "👍" if weight > 0 else "👎"
            print(f"  {sentiment} {feature_name}: {weight:+.2f}")

        print("\n" + "="*60 + "\n")


def main():
    """Main CLI entry point"""
    # Paths
    models_dir = Path('output/models')
    movies_path = Path('output/processed/movies.parquet')
    keyword_db_path = Path('output/models/keyword_database.pkl')
    storage_dir = Path('output/interactive_learning')

    print("Loading recommendation system...")

    # Load SmartEngine
    smart_engine = load_smart_system(models_dir, movies_path, keyword_db_path)

    # Load movies
    movies_df = pd.read_parquet(movies_path)

    # Initialize components
    encoder = FeatureEncoder(movies_df)

    # Load or create learning models
    storage = LearningStorage(storage_dir)
    sequence_learner, global_learner = storage.load_models()

    # Create session learner (fresh per session)
    session_learner = SessionLearner(feature_names=encoder.feature_names)

    # Create selector
    selector = InteractiveSelector(
        encoder=encoder,
        movies_df=movies_df,
        session_learner=session_learner,
        sequence_learner=sequence_learner,
        global_learner=global_learner
    )

    print("✓ System loaded\n")

    # Run session
    session = InteractiveSession(
        smart_engine=smart_engine,
        encoder=encoder,
        selector=selector,
        storage=storage,
        user_id=0  # Demo user
    )

    session.run_session()

    # Save learning models
    print("Saving learning models...")
    storage.save_models(sequence_learner, global_learner)
    print("✓ Models saved\n")

    print("Thank you for using the interactive movie recommender!")


if __name__ == '__main__':
    main()
