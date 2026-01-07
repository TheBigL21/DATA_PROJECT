"""
INTERACTIVE MOVIE FINDER

Full workflow:
1. Answer 4 questions (evening type, genres, era, themes) - same as movie_finder
2. System generates 100 candidates using SmartEngine.recommend()
3. Selects the BEST movie from 100 as first choice (highest score)
4. Interactive selection: shows one movie at a time
5. User inputs: 0 (yes - show similar), 1 (no - show different), 2 (final choice)
6. System learns and adapts pool in real-time
7. On final choice: shows complete movie details
Run this file to experience the interactive system!
"""

from pathlib import Path
import sys
import uuid
from datetime import datetime
import json
import logging

# Compute repo root (this file is in src/, so go up one level)
REPO_ROOT = Path(__file__).parent.parent

# Add repo root to Python path so imports work when run directly
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.recommendation.smart_engine import load_smart_system
from src.recommendation.interactive_selector import (
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
from src.recommendation.keyword_recommender import KeywordRecommender
from src.recommendation.time_period_filter import TimePeriodFilter
from src.recommendation.source_material_filter import SourceMaterialFilter
from src.auth.user_auth import authenticate
from src.analytics.genre_tracker import GenreAnalytics


# ============================================================================
# CONSTANTS
# ============================================================================

SEPARATOR_WIDE = 80
SEPARATOR_NARROW = 60

# Evening types configuration
EVENING_TYPES = [
    "Chill Evening by myself",
    "Date night",
    "Family night",
    "Friends night"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_screen():
    import os
    os.system('clear' if os.name != 'nt' else 'cls')


def load_genre_config():
    """Load genre allocation config"""
    config_path = REPO_ROOT / 'src' / 'config' / 'genre_allocation.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def print_header():
    """Print welcome header"""
    print("\n" + "="*SEPARATOR_WIDE)
    print(" " * 20 + "INTERACTIVE MOVIE FINDER")
    print("="*SEPARATOR_WIDE)
    print("\nWelcome! This system will help you find the perfect movie.")
    print("It learns from your choices and adapts in real-time.\n")
    print("="*SEPARATOR_WIDE + "\n")


def ask_evening_type():
    """Q1: Evening Setup"""
    clear_screen()
    print("\n" + "="*SEPARATOR_NARROW)
    print("  INTERACTIVE MOVIE FINDER - Find Your Perfect Movie Tonight")
    print("="*SEPARATOR_NARROW + "\n")

    print("What's your plan for tonight?\n")
    for i, evening_type in enumerate(EVENING_TYPES, 1):
        print(f"  {i}. {evening_type}")

    while True:
        choice = input(f"\nSelect (1-{len(EVENING_TYPES)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(EVENING_TYPES):
            return EVENING_TYPES[int(choice) - 1]
        print(f"Invalid choice. Please enter 1-{len(EVENING_TYPES)}.")


def ask_genres(evening_type, genre_config):
    """Q2: Genre Selection (1-2 genres)"""
    clear_screen()
    print("\n" + "="*SEPARATOR_NARROW)
    print(f"  EVENING TYPE: {evening_type}")
    print("="*SEPARATOR_NARROW + "\n")

    print("What genre(s) match your mood? (Select 1-2)\n")

    core_genres = genre_config[evening_type]['core']
    extended_genres = genre_config[evening_type]['extended']

    # Display core genres
    print("POPULAR CHOICES:")
    for i, genre in enumerate(core_genres, 1):
        print(f"  {i}. {genre.capitalize()}")

    print(f"\n  {len(core_genres) + 1}. More genres...")

    # Get user selections
    selected_genres = []
    show_extended = False

    while len(selected_genres) < 2:
        if show_extended:
            print("\n\nMORE GENRES:")
            for i, genre in enumerate(extended_genres, 1):
                print(f"  {i + len(core_genres)}. {genre.capitalize()}")

        prompt = "\nSelect genre" if len(selected_genres) == 0 else "\nSelect another genre (or press Enter to continue)"
        choice = input(f"{prompt}: ").strip()

        if not choice and len(selected_genres) >= 1:
            break

        if not choice.isdigit():
            continue

        choice_num = int(choice)

        # Check for "More genres"
        if choice_num == len(core_genres) + 1 and not show_extended:
            show_extended = True
            continue

        # Get selected genre
        if 1 <= choice_num <= len(core_genres):
            genre = core_genres[choice_num - 1]
        elif show_extended and len(core_genres) < choice_num <= len(core_genres) + len(extended_genres):
            genre = extended_genres[choice_num - len(core_genres) - 1]
        else:
            print("Invalid choice")
            continue

        if genre not in selected_genres:
            selected_genres.append(genre)
            print(f"Selected: {genre.capitalize()}")

            if len(selected_genres) == 2:
                break

    return selected_genres, core_genres + (extended_genres if show_extended else [])


def ask_era():
    """Q3: Time Period/Era Selection"""
    print("\n" + "-"*60)
    print("What time period are you in the mood for?\n")

    era_options = TimePeriodFilter.get_era_options()

    # Display all 5 era options
    for i, era in enumerate(era_options, 1):
        print(f"  {i}. {era['label']}")

    while True:
        choice = input("\nSelect era (1-5): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(era_options):
            selected_era = era_options[int(choice) - 1]
            print(f"Selected: {selected_era['label']}")
            return selected_era['id']

        print("Invalid choice. Please enter a number 1-5.")


def get_relevant_source_material(selected_genres):
    """Determine the most relevant source material option for given genres"""
    # Genre to source material mapping (most relevant for each genre)
    genre_source_map = {
        'fantasy': 'book',
        'sci-fi': 'book',
        'romance': 'book',
        'drama': 'true_story',
        'biography': 'true_story',
        'history': 'true_story',
        'war': 'true_story',
        'action': 'comic',
        'adventure': 'book',
        'crime': 'true_story',
        'thriller': 'book',
        'mystery': 'book',
        'horror': 'book',
        'musical': 'play_musical',
        'comedy': 'book'
    }

    # Check selected genres and find most common source material
    source_votes = {}
    for genre in selected_genres:
        genre_lower = genre.lower()
        if genre_lower in genre_source_map:
            source = genre_source_map[genre_lower]
            source_votes[source] = source_votes.get(source, 0) + 1

    # Return most voted source, or default to 'book'
    if source_votes:
        most_relevant = max(source_votes.items(), key=lambda x: x[1])[0]
        return most_relevant

    return 'book'  # Default fallback


def ask_thematic_keywords(selected_genres, keyword_recommender):
    """Q4: Thematic Keywords Selection (with optional source material)"""
    print("\n" + "-"*60)
    print("What themes interest you? (Select up to 3, or press Enter to skip)\n")

    # Get thematic keywords (now filtered via KeywordFilter)
    thematic_keywords = keyword_recommender.get_keywords_for_genres(selected_genres, num_keywords=8)

    if not thematic_keywords:
        print("No specific keywords available for these genres.")
        return [], 'any'

    print("THEMES:")
    for i, kw in enumerate(thematic_keywords, 1):
        kw_display = kw.replace('_', ' ').title()
        print(f"  {i}. {kw_display}")

    # Add ONE relevant source material option
    relevant_source = get_relevant_source_material(selected_genres)
    source_info = SourceMaterialFilter.SOURCE_KEYWORDS[relevant_source]
    source_display = source_info['label']

    print(f"\n  {len(thematic_keywords) + 1}. {source_display}")

    print("\n" + "-"*60)

    selected = []
    selected_source = None

    while len(selected) < 3:
        if len(selected) == 0:
            prompt = "\nSelect a theme (or press Enter to skip)"
        else:
            prompt = f"\nSelect another theme (or press Enter to continue) [{len(selected)}/3]"

        choice = input(f"{prompt}: ").strip()

        if not choice:
            break

        if not choice.isdigit():
            continue

        choice_num = int(choice)

        # Check if it's the source material option
        if choice_num == len(thematic_keywords) + 1:
            if selected_source:
                print(f"Source material already selected: {source_display}")
                continue

            selected_source = relevant_source
            print(f"Selected: {source_display}")

        elif 1 <= choice_num <= len(thematic_keywords):
            keyword = thematic_keywords[choice_num - 1]

            if keyword in selected:
                print(f"Already selected: {keyword}")
                continue

            selected.append(keyword)
            kw_display = keyword.replace('_', ' ').title()
            print(f"Selected: {kw_display}")

            if len(selected) == 3:
                print("\nMaximum of 3 themes selected.")
                break
        else:
            print("Invalid choice")

    if selected:
        print("\nThemes selected:")
        for kw in selected:
            print(f"  - {kw.replace('_', ' ').title()}")
    if selected_source:
        print(f"  - {source_display}")

    return selected, selected_source if selected_source else 'any'


def run_interactive_selection(
    smart_engine,
    encoder,
    selector,
    pool_manager,
    context,
    user_id,
    session_id,
    storage,
    initial_candidates
):
    """
    Run the interactive movie selection loop.

    First movie shown is the BEST from initial_candidates (top-ranked by SmartEngine).
    Subsequent movies are selected by interactive selector based on user feedback.

    Args:
        smart_engine: SmartEngine instance
        encoder: FeatureEncoder
        selector: InteractiveSelector
        pool_manager: AdaptivePoolManager
        context: User context dict
        user_id: User ID
        session_id: Session ID
        storage: LearningStorage
        initial_candidates: List of movie IDs ranked by SmartEngine (best first)

    Returns:
        tuple: (final_movie_id, session_history) or (None, history) if no final choice
    """
    print("="*SEPARATOR_WIDE)
    print("INTERACTIVE SELECTION")
    print("="*SEPARATOR_WIDE)
    print("\nI'll show you movies one at a time. For each movie, choose:")
    print()
    print("  0 or Enter  -> YES (I like this, show me similar movies)")
    print("  1           -> NO (not interested, show me something different)")
    print("  2           -> FINAL (this is the one I want to watch!)")
    print()
    print("The system learns from each choice and adapts the recommendations.")
    print("="*SEPARATOR_WIDE + "\n")

    input("Press Enter to start...")
    print()

    position = 1
    last_movie_id = None
    session_history = []

    while True:
        # For first movie, select the BEST from initial candidates
        if position == 1 and initial_candidates:
            movie_id = initial_candidates[0]  # Best movie from SmartEngine
            print("[Showing top recommendation from your preferences]\n")
        else:
            # For subsequent movies, use interactive selector
            movie_id = selector.select_next(
                pool_manager=pool_manager,
                context=context,
                last_movie_id=last_movie_id
            )

        if movie_id is None:
            print("\n" + "="*SEPARATOR_WIDE)
            print("NO MORE CANDIDATES AVAILABLE")
            print("="*SEPARATOR_WIDE)
            print("We've run out of movies matching your preferences.")
            print("This can happen if you've been very selective.")
            print()

            if session_history:
                print("Movies you liked this session:")
                for item in session_history:
                    if item['action'] in ['yes', 'final']:
                        m = smart_engine.movies[smart_engine.movies['movie_id'] == item['movie_id']].iloc[0]
                        print(f"  - {m['title']} ({m['year']})")

            print("\nSession ended without final selection.")
            return None, session_history

        # Get movie details
        movie = smart_engine.movies[
            smart_engine.movies['movie_id'] == movie_id
        ].iloc[0]

        # Display movie card
        display_movie_card(movie, position)

        # Get user input
        while True:
            user_input = input("Your choice: ").strip()

            if user_input == '' or user_input == '0':
                action = 'yes'
                break
            elif user_input == '1':
                action = 'no'
                break
            elif user_input == '2':
                action = 'final'
                print("-> FINAL CHOICE - Perfect!\n")
                break
            else:
                print("Invalid input. Please enter 0 (yes), 1 (no), or 2 (final).")

        # Mark as shown
        pool_manager.mark_shown(movie_id)

        # Process feedback
        selector.process_feedback(
            movie_id=movie_id,
            action=action,
            context=context,
            pool_manager=pool_manager,
            last_movie_id=last_movie_id
        )

        # Update persistent feedback learner (NEW: integrates with ML system)
        try:
            smart_engine.update_feedback(
                user_id=user_id,
                session_id=session_id,
                movie_id=movie_id,
                action=action,
                context=context,
                position_in_session=position,
                previous_movie_id=last_movie_id
            )
        except Exception as e:
            # Log but don't fail if feedback update fails
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update persistent feedback: {e}")

        # Display feedback message after processing
        if action == 'yes':
            print("Searching similar movies...\n")
        elif action == 'no':
            print("Searching different movies...\n")

        # Log event (for interactive selector's own learning)
        event = FeedbackEvent(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            context=context,
            movie_id=movie_id,
            action=action,
            satisfaction=action_to_satisfaction(action),
            position_in_session=position,
            previous_movie_id=last_movie_id
        )
        storage.log_event(event)

        # Track history (format matches what recommend() expects)
        session_history.append({
            'movie_id': movie_id,
            'action': action,
            'position': position
        })

        # Update for next iteration
        last_movie_id = movie_id
        position += 1

        # If final choice, display details and end
        if action == 'final':
            display_movie_details(movie)
            return movie_id, session_history

        # Safety: max 25 movies per session
        if position > 25:
            print("\n" + "="*SEPARATOR_WIDE)
            print("MAXIMUM SESSION LENGTH REACHED")
            print("="*SEPARATOR_WIDE)
            print("You've reviewed 25 movies. Let's wrap up this session.\n")
            break

    return None, session_history


def print_session_summary(selector, session_history):
    """Print session summary with learned preferences"""
    print("\n" + "="*SEPARATOR_WIDE)
    print("SESSION SUMMARY")
    print("="*SEPARATOR_WIDE)

    # Count actions
    yes_count = sum(1 for h in session_history if h['action'] == 'yes')
    no_count = sum(1 for h in session_history if h['action'] == 'no')
    final_count = sum(1 for h in session_history if h['action'] == 'final')

    print(f"\nMovies reviewed: {len(session_history)}")
    print(f"  - YES (liked): {yes_count}")
    print(f"  - NO (rejected): {no_count}")
    print(f"  - FINAL (selected): {final_count}")

    # Show learned preferences
    if selector.session_learner.n_updates > 0:
        print("\nWhat the system learned about your preferences:")
        print("-" * 60)

        top_features = selector.session_learner.get_top_features(top_k=8)

        positive = [(name, weight) for name, weight in top_features if weight > 0]
        negative = [(name, weight) for name, weight in top_features if weight < 0]

        if positive:
            print("\nYou LIKE:")
            for feature, weight in positive[:5]:
                feature_display = feature.replace('_', ' ').title()
                print(f"  + {feature_display} (strength: {weight:+.2f})")

        if negative:
            print("\nYou DISLIKE:")
            for feature, weight in negative[:5]:
                feature_display = feature.replace('_', ' ').title()
                print(f"  - {feature_display} (strength: {weight:+.2f})")

    print("\n" + "="*SEPARATOR_WIDE)


def main():
    """Main entry point for interactive movie finder"""

    # Print welcome
    print_header()

    # Configuration - Use absolute paths relative to repo root
    models_dir = REPO_ROOT / 'data' / 'models'
    movies_path = REPO_ROOT / 'data' / 'processed' / 'movies.parquet'
    keyword_db_path = REPO_ROOT / 'data' / 'models' / 'keyword_database.pkl'
    feedback_db_path = REPO_ROOT / 'data' / 'feedback.db'
    storage_dir = REPO_ROOT / 'data' / 'interactive_learning'

    user_id = 0  # Demo user ID (in production, use actual user ID)
    session_id = str(uuid.uuid4())[:8]

    # Load system
    print("Loading recommendation system (this may take 10-15 seconds)...")
    print("  - Loading collaborative filtering model...")
    print("  - Loading co-occurrence graph...")
    print("  - Loading movie database...")
    print("  - Loading feedback learning system...")

    try:
        smart_engine = load_smart_system(
            models_dir,
            movies_path,
            keyword_db_path if keyword_db_path.exists() else None,
            feedback_db_path=feedback_db_path
        )
        movies_df = pd.read_parquet(movies_path)
        genre_config = load_genre_config()
        keyword_recommender = KeywordRecommender(str(movies_path))
        analytics = GenreAnalytics()

        print(f"\nSystem loaded successfully!")
        print(f"{len(movies_df)} movies available\n")

    except Exception as e:
        print(f"\nError loading system: {e}")
        print("\nPlease ensure:")
        print("  1. Models exist in output/models/")
        print("  2. Movies parquet exists in output/processed/")
        print("\nRun the training pipeline first if needed.")
        return 1

    # Initialize interactive components
    print("Initializing interactive selector...")

    encoder = FeatureEncoder(movies_df)
    storage = LearningStorage(storage_dir)

    # Load persistent learning models
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

    print("Interactive selector ready\n")

    # Step 1: Ask 4 questions (same as movie_finder)
    evening_type = ask_evening_type()
    selected_genres, presented_genres = ask_genres(evening_type, genre_config)
    selected_themes, source_material = ask_thematic_keywords(selected_genres, keyword_recommender)
    selected_era = ask_era()

    # Log for analytics
    analytics.log_presentation(evening_type, presented_genres)
    analytics.log_selection(evening_type, selected_genres)

    # Build context
    context = {
        'genres': selected_genres,
        'era': selected_era,
        'source_material': source_material,
        'themes': selected_themes
    }

    # Step 2: Generate 100 candidates using SmartEngine (same as movie_finder)
    try:
        initial_candidates = smart_engine.recommend(
            user_id=user_id,
            genres=context['genres'],
            era=context['era'],
            source_material=context['source_material'],
            themes=context['themes'],
            session_history=[],
            session_id=session_id,
            top_k=100
        )

    except Exception as e:
        print(f"Error generating candidates: {e}")
        print("\nTry different preferences or check your filters.")
        return 1

    # Step 3: Create adaptive pool manager
    pool_manager = AdaptivePoolManager(
        encoder=encoder,
        movies_df=movies_df,
        initial_candidates=initial_candidates,
        context=context,
        pool_size_target=80
    )

    # Step 4: Run interactive selection (first movie is BEST from 100 candidates)
    final_movie_id, session_history = run_interactive_selection(
        smart_engine=smart_engine,
        encoder=encoder,
        selector=selector,
        pool_manager=pool_manager,
        context=context,
        user_id=user_id,
        session_id=session_id,
        storage=storage,
        initial_candidates=initial_candidates
    )

    # Step 5: Session summary
    if session_history:
        print_session_summary(selector, session_history)

    # Step 6: Save learning models
    if session_history:
        print("\nSaving learning models for future sessions...")
        storage.save_models(sequence_learner, global_learner)
        print("Models saved\n")

    # Farewell
    print("="*SEPARATOR_WIDE)
    print("Thank you for using the Interactive Movie Finder!")
    print("="*SEPARATOR_WIDE)

    if final_movie_id:
        print("\nEnjoy your movie!")
    else:
        print("\nFeel free to run again with different preferences!")

    print()

    return 0


if __name__ == '__main__':
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Goodbye!")
        exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
