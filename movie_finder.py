"""
MOVIE FINDER - 3-Question Recommendation Interface (Enhanced)

Updated with inference-based keyword system
"""

from pathlib import Path
import sys
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))


from recommendation.smart_engine import load_smart_system
from analytics.genre_tracker import GenreAnalytics
from auth.user_auth import authenticate
from recommendation.keyword_recommender import KeywordRecommender
from recommendation.time_period_filter import TimePeriodFilter
from recommendation.source_material_filter import SourceMaterialFilter


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
            print(f"Selected: {genre.capitalize()}")

            if len(selected_genres) == 2:
                break

    return selected_genres, core_genres + (extended_genres if show_extended else [])


def ask_era():
    """Q3: Time Period/Era Selection"""
    print("\n" + "-"*60)
    print("What vibe are you feeling?\n")

    era_options = TimePeriodFilter.get_era_options()

    # Display only the label without description
    print("  1. New Generation (2010+)")
    print("  2. Golden Era (1990-2009)")
    print("  3. Old School (Before 1990)")
    print("  4. Doesn't matter")

    while True:
        choice = input("\nSelect vibe (1-4): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(era_options):
            selected_era = era_options[int(choice) - 1]
            print(f"Selected: {selected_era['label']}")
            return selected_era['id']

        print("Invalid choice. Please enter a number 1-4.")




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
        return []

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
            selected.append(f"__SOURCE__{relevant_source}")  # Mark as source material

            if len(selected) == 3:
                print("\nMaximum of 3 themes selected.")
                break

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
            if kw.startswith("__SOURCE__"):
                print(f"  - {source_display}")
            else:
                print(f"  - {kw.replace('_', ' ').title()}")

    return selected


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
        print(f"Region: {region_info}")

    # Display search summary
    print(f"\nSearching for {', '.join([g.capitalize() for g in user_answers['genres']])} movies...")

    if user_answers.get('era') and user_answers['era'] != 'any':
        era_label = TimePeriodFilter.get_era_short_label(user_answers['era'])
        print(f"  - Era: {era_label}")

    if user_answers.get('source_material') and user_answers['source_material'] != 'any':
        source_label = SourceMaterialFilter.get_source_short_label(user_answers['source_material'])
        print(f"  - Source: {source_label}")

    if user_answers.get('themes') and len(user_answers['themes']) > 0:
        themes_str = ', '.join([t.replace('_', ' ').title() for t in user_answers['themes']])
        print(f"  - Themes: {themes_str}")

    print()

    # Generate recommendations
    try:
        recommendations = engine.recommend(
            user_id=0,
            genres=user_answers['genres'],
            era=user_answers.get('era'),
            source_material=user_answers.get('source_material'),
            themes=user_answers.get('themes', []),
            session_history=[],
            top_k=10
        )

        print("="*60)
        print("TOP 10 MOVIES FOR YOU:")
        print("="*60 + "\n")

        for rank, movie_id in enumerate(recommendations, 1):
            movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0].to_dict()
            genres_str = ', '.join([g.capitalize() for g in list(movie['genres'])[:3]])

            print(f"{rank:2d}. {movie['title']} ({movie['year']})")
            print(f"    {genres_str} | {movie['runtime']:.0f} min | Rating: {movie['avg_rating']:.1f}/10")
            print(f"    {movie['num_votes']:,} votes | Director: {movie['director']}")

            # Display TMDB information if available
            if pd.notna(movie.get('tmdb_id')):
                # Show description
                if pd.notna(movie.get('description')) and movie['description']:
                    desc = movie['description']
                    # Truncate description to 150 chars
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    print(f"    {desc}")

                # Show poster URL
                if pd.notna(movie.get('poster_url')) and movie['poster_url']:
                    print(f"    Poster: {movie['poster_url']}")

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
        keyword_db_path = Path('./output/models/keyword_database.pkl')
        engine = load_smart_system(models_dir, movies_path, keyword_db_path)
        genre_config = load_genre_config()
        analytics = GenreAnalytics()
        keyword_recommender = KeywordRecommender(str(movies_path))

    except Exception as e:
        print(f"\nError: Could not load recommendation system")
        print(f"Details: {e}")
        print("\nPlease ensure the pipeline has completed successfully.")
        return

    # STEP 3: Ask 4 questions (source material now integrated into keywords)
    evening_type = ask_evening_type()
    selected_genres, presented_genres = ask_genres(evening_type, genre_config)
    selected_era = ask_era()
    selected_themes = ask_thematic_keywords(selected_genres, keyword_recommender)

    # Log for ML learning
    analytics.log_presentation(evening_type, presented_genres)
    analytics.log_selection(evening_type, selected_genres)

    # Extract source material from themes (if selected)
    source_material = 'any'
    themes_only = []

    for theme in selected_themes:
        if theme.startswith("__SOURCE__"):
            source_material = theme.replace("__SOURCE__", "")
        else:
            themes_only.append(theme)

    # Store answers
    user_answers = {
        'evening_type': evening_type,
        'genres': selected_genres,
        'era': selected_era,
        'source_material': source_material,
        'themes': themes_only
    }

    # STEP 4: Generate and display recommendations
    success = display_recommendations(engine, user_answers, user_data)

    if success:
        print("\n" + "="*60)
        print("Recommendations generated successfully!")
        print("="*60 + "\n")

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
