"""
MOVIE FINDER - 3-Question Recommendation Interface

Complete interface with ML-powered genre learning
"""

from pathlib import Path
import sys
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))


from recommendation.enhanced_engine import load_system
from analytics.genre_tracker import GenreAnalytics
from auth.user_auth import authenticate
from recommendation.keyword_recommender import KeywordRecommender


def clear_screen():
    import os
    os.system('clear' if os.name != 'nt' else 'cls')


def load_genre_config():
    """Load genre allocation config"""
    with open('config/genre_allocation.json', 'r') as f:
        return json.load(f)


def ask_evening_type():
    """Q1: Evening Setup"""
    clear_screen()
    print("\n" + "="*60)
    print("  MOVIE FINDER - Find Your Perfect Movie Tonight")
    print("="*60 + "\n")

    print("What's your plan for tonight?\n")
    options = [
        "1. Chill Evening by myself",
        "2. Date night",
        "3. Family night",
        "4. Friends night"
    ]

    for opt in options:
        print(f"  {opt}")

    while True:
        choice = input("\nSelect (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            evening_types = [
                "Chill Evening by myself",
                "Date night",
                "Family night",
                "Friends night"
            ]
            return evening_types[int(choice) - 1]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def ask_genres(evening_type, genre_config):
    """Q2: Genre Selection (1-2 genres)"""
    clear_screen()
    print("\n" + "="*60)
    print(f"  EVENING TYPE: {evening_type}")
    print("="*60 + "\n")

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
            print(f"✓ Selected: {genre.capitalize()}")

            if len(selected_genres) == 2:
                break

    return selected_genres, core_genres + (extended_genres if show_extended else [])


def ask_age_preference():
    """Q3: Age of Movie"""
    print("\n" + "-"*60)
    print("What era are you in the mood for?\n")

    options = [
        "1. Less than 5 years old",
        "2. Less than 10 years old",
        "3. Less than 20 years old",
        "4. Doesn't Matter"
    ]

    for opt in options:
        print(f"  {opt}")

    while True:
        choice = input("\nSelect (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            ages = [
                "Less than 5 years old",
                "Less than 10 years old",
                "Less than 20 years old",
                "Doesn't Matter"
            ]
            return ages[int(choice) - 1]
        print("Invalid choice")


def ask_keywords(selected_genres, keyword_recommender):
    """Q3: Keyword Selection"""
    print("\n" + "-"*60)
    print("Select keywords to refine your search (Select up to 3)\n")

    # Get recommended keywords based on genres
    recommended = keyword_recommender.get_keywords_for_genres(selected_genres, num_keywords=6)

    print("SUGGESTED KEYWORDS:")
    for i, keyword in enumerate(recommended, 1):
        print(f"  {i}. {keyword.capitalize()}")

    print(f"\n  {len(recommended) + 1}. Skip (no keywords)")

    selected_keywords = []
    while len(selected_keywords) < 3:
        prompt = "\nSelect keyword" if len(selected_keywords) == 0 else "\nSelect another keyword (or press Enter to continue)"
        choice = input(f"{prompt}: ").strip()

        if not choice and len(selected_keywords) >= 0:
            break

        if not choice.isdigit():
            continue

        choice_num = int(choice)

        # Check for skip
        if choice_num == len(recommended) + 1:
            break

        # Get selected keyword
        if 1 <= choice_num <= len(recommended):
            keyword = recommended[choice_num - 1]
            if keyword not in selected_keywords:
                selected_keywords.append(keyword)
                print(f"✓ Selected: {keyword.capitalize()}")

                if len(selected_keywords) == 3:
                    break
        else:
            print("Invalid choice")

    return selected_keywords


def ask_quality():
    """Q4: Quality Level"""
    print("\n" + "-"*60)
    print("What quality level?\n")

    options = [
        "1. Only the best (8.0+)",
        "2. Good movies (7.0+)",
        "3. Hidden gems (6.0+)",
        "4. I'll try anything (5.0+)"
    ]

    for opt in options:
        print(f"  {opt}")

    while True:
        choice = input("\nSelect (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            qualities = [
                "Only the best (8.0+)",
                "Good movies (7.0+)",
                "Hidden gems (6.0+)",
                "I'll try anything (5.0+)"
            ]
            return qualities[int(choice) - 1]
        print("Invalid choice")


def display_recommendations(engine, user_answers, user_data):
    """Generate and display recommendations"""
    clear_screen()
    print("\n" + "="*60)
    print("  YOUR PERSONALIZED RECOMMENDATIONS")
    print("="*60)

    # Show user info and region
    if user_data['name'] != 'Guest':
        print(f"\nUser: {user_data['name']} | Region: {user_data['region']}")
        from recommendation.region_weighting import RegionWeighting
        weighter = RegionWeighting()
        region_info = weighter.get_region_info(user_data['region'])
        print(f"📍 {region_info}")

    print(f"\nSearching for {', '.join([g.capitalize() for g in user_answers['genres']])} movies...")
    if user_answers.get('keywords'):
        print(f"Keywords: {', '.join([k.capitalize() for k in user_answers['keywords']])}")
    print(f"Era: {user_answers['age']}")
    print(f"Quality: {user_answers['quality']}\n")

    # Generate recommendations
    try:
        recommendations = engine.recommend(
            user_id=0,
            evening_type=user_answers['evening_type'],
            selected_genres=user_answers['genres'],
            age_preference=user_answers['age'],
            runtime_pref="Doesn't matter",
            quality_level=user_answers['quality'],
            popularity_level="Mix of everything",
            session_history=[],
            user_region=user_data.get('region', 'Other'),  # Pass user region
            selected_keywords=user_answers.get('keywords', []),  # Pass keywords
            top_k=10
        )

        print("="*60)
        print("TOP 10 MOVIES FOR YOU:")
        print("="*60 + "\n")

        for rank, movie_id in enumerate(recommendations, 1):
            movie = engine.base_engine.movie_dict[movie_id]
            genres_str = ', '.join([g.capitalize() for g in list(movie['genres'])[:3]])

            print(f"{rank:2d}. {movie['title']} ({movie['year']})")
            print(f"    {genres_str} | {movie['runtime']:.0f} min | ⭐ {movie['avg_rating']:.1f}/10")
            print(f"    👥 {movie['num_votes']:,} votes | Director: {movie['director']}")

            # Display TMDB information if available
            if pd.notna(movie.get('tmdb_id')):
                # Show composite rating if different from IMDb
                if pd.notna(movie.get('composite_rating')) and abs(movie['composite_rating'] - movie['avg_rating']) > 0.1:
                    print(f"    🎬 Composite Rating: {movie['composite_rating']:.1f}/10 (IMDb + TMDB)")

                # Show description
                if pd.notna(movie.get('description')) and movie['description']:
                    desc = movie['description']
                    # Truncate description to 150 chars
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    print(f"    📝 {desc}")

                # Show poster URL
                if pd.notna(movie.get('poster_url')) and movie['poster_url']:
                    print(f"    🖼️  Poster: {movie['poster_url']}")

            print()

        return True

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution"""

    # STEP 1: Authenticate user (Login/Signup/Skip)
    print("\n" + "="*60)
    print("  WELCOME TO MOVIE FINDER")
    print("="*60)
    user_data = authenticate()

    # STEP 2: Load recommendation systems
    print("\nInitializing Movie Finder...")
    try:
        models_dir = Path('./output/models')
        movies_path = Path('./output/processed/movies.parquet')
        engine = load_system(models_dir, movies_path)
        genre_config = load_genre_config()
        analytics = GenreAnalytics()
        keyword_recommender = KeywordRecommender(str(movies_path))

    except Exception as e:
        print(f"\nError: Could not load recommendation system")
        print(f"Details: {e}")
        print("\nPlease ensure the pipeline has completed successfully.")
        return

    # STEP 3: Ask all questions
    evening_type = ask_evening_type()
    selected_genres, presented_genres = ask_genres(evening_type, genre_config)
    selected_keywords = ask_keywords(selected_genres, keyword_recommender)
    age_pref = ask_age_preference()
    quality = ask_quality()

    # Log for ML learning
    analytics.log_presentation(evening_type, presented_genres)
    analytics.log_selection(evening_type, selected_genres)

    # Store answers
    user_answers = {
        'evening_type': evening_type,
        'genres': selected_genres,
        'keywords': selected_keywords,
        'age': age_pref,
        'quality': quality
    }

    # STEP 4: Generate and display recommendations (with region-based weighting)
    success = display_recommendations(engine, user_answers, user_data)

    if success:
        print("\n" + "="*60)
        print("✓ Recommendations generated successfully!")
        print("="*60 + "\n")

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
    
