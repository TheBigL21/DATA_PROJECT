"""
TIME PERIOD FILTER MODULE

Handles time period/era filtering for movie recommendations.
Maps user-friendly era labels to year ranges.
"""

from typing import Tuple, Optional


class TimePeriodFilter:
    """Manages time period filtering for movie recommendations"""

    # Era definitions with year ranges
    ERAS = {
        'new_era': {
            'label': 'New Era (2010+)',
            'short_label': 'New Era',
            'year_min': 2010,
            'year_max': 2030,
            'description': 'Contemporary cinema from the last decade'
        },
        'golden_era': {
            'label': 'Golden Era (1990-2010)',
            'short_label': 'Golden Era',
            'year_min': 1990,
            'year_max': 2010,
            'description': 'Two decades of iconic modern cinema'
        },
        'millennium': {
            'label': 'The 2nd Millennium (2000+)',
            'short_label': '2nd Millennium',
            'year_min': 2000,
            'year_max': 2030,
            'description': '21st century films'
        },
        'old_school': {
            'label': 'Old School (Before 2000)',
            'short_label': 'Old School',
            'year_min': 1900,
            'year_max': 1999,
            'description': 'Classic cinema from the 20th century'
        },
        'any': {
            'label': "Doesn't matter",
            'short_label': 'Any Era',
            'year_min': None,
            'year_max': None,
            'description': 'No preference on time period'
        }
    }

    # Backward compatibility mapping (old IDs → new IDs)
    BACKWARD_COMPATIBLE_ERAS = {
        'new_gen': 'new_era',
        'recent': 'new_era',
        'modern': 'millennium',
        'throwback': 'golden_era'
    }

    @classmethod
    def get_era_options(cls) -> list:
        """
        Get list of era options for display (5 options).

        Returns:
            List of dicts with era info
        """
        # Return the 5 era options
        main_eras = ['new_era', 'golden_era', 'millennium', 'old_school', 'any']
        return [
            {
                'id': era_id,
                'label': cls.ERAS[era_id]['label'],
                'short_label': cls.ERAS[era_id]['short_label'],
                'description': cls.ERAS[era_id]['description']
            }
            for era_id in main_eras
        ]

    @classmethod
    def get_year_range(cls, era_id: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get year range for an era.

        Args:
            era_id: Era identifier (e.g., 'recent', 'modern', 'any')

        Returns:
            Tuple of (min_year, max_year). Returns (None, None) for 'any'
        """
        # Map old era IDs to new ones for backward compatibility
        if era_id in cls.BACKWARD_COMPATIBLE_ERAS:
            era_id = cls.BACKWARD_COMPATIBLE_ERAS[era_id]

        if era_id not in cls.ERAS:
            return None, None

        era = cls.ERAS[era_id]
        return era['year_min'], era['year_max']

    @classmethod
    def calculate_era_score(cls, movie_year: int, selected_era: str) -> float:
        """
        Calculate how well a movie matches the selected era.

        Args:
            movie_year: Year the movie was released
            selected_era: Selected era ID

        Returns:
            Score between 0.0 and 1.0
        """
        if selected_era == 'any' or selected_era not in cls.ERAS:
            return 1.0  # No preference, all movies score equally

        year_min, year_max = cls.get_year_range(selected_era)

        if year_min is None or year_max is None:
            return 1.0

        # Perfect match if within range
        if year_min <= movie_year <= year_max:
            return 1.0

        # Gradual decay outside the range
        if movie_year < year_min:
            years_before = year_min - movie_year
            # Decay: -5 years = 0.8, -10 years = 0.6, -20 years = 0.4
            decay = max(0.0, 1.0 - (years_before / 25.0))
            return decay

        if movie_year > year_max:
            years_after = movie_year - year_max
            # Decay: +5 years = 0.8, +10 years = 0.6, +20 years = 0.4
            decay = max(0.0, 1.0 - (years_after / 25.0))
            return decay

        return 1.0

    @classmethod
    def filter_movies_by_era(cls, movies: list, selected_era: str, strict: bool = False) -> list:
        """
        Filter movies by era.

        Args:
            movies: List of movie dicts (must have 'year' field)
            selected_era: Selected era ID
            strict: If True, only return movies within range. If False, score all movies.

        Returns:
            Filtered/scored list of movies
        """
        if selected_era == 'any':
            return movies

        year_min, year_max = cls.get_year_range(selected_era)

        if year_min is None or year_max is None:
            return movies

        if strict:
            # Hard filter: only movies within range
            return [
                movie for movie in movies
                if 'year' in movie and year_min <= movie['year'] <= year_max
            ]
        else:
            # Soft filter: score all movies
            for movie in movies:
                if 'year' in movie:
                    era_score = cls.calculate_era_score(movie['year'], selected_era)
                    movie['era_score'] = era_score
            return movies

    @classmethod
    def get_era_label(cls, era_id: str) -> str:
        """Get display label for an era"""
        if era_id in cls.ERAS:
            return cls.ERAS[era_id]['label']
        return "Unknown Era"

    @classmethod
    def get_era_short_label(cls, era_id: str) -> str:
        """Get short display label for an era"""
        if era_id in cls.ERAS:
            return cls.ERAS[era_id]['short_label']
        return "Unknown"


if __name__ == '__main__':
    """Test the time period filter"""
    print("="*70)
    print("TIME PERIOD FILTER TEST")
    print("="*70)

    # Display all era options
    print("\nAvailable Eras:\n")
    for i, era in enumerate(TimePeriodFilter.get_era_options(), 1):
        year_min, year_max = TimePeriodFilter.get_year_range(era['id'])
        year_range = f"{year_min}-{year_max}" if year_min else "Any"
        print(f"  {i}. {era['label']}")
        print(f"     Range: {year_range}")
        print(f"     {era['description']}\n")

    # Test era scoring
    print(f"{'='*70}")
    print("ERA SCORING TEST")
    print(f"{'='*70}\n")

    test_cases = [
        ('recent', 2020, 1.0),
        ('recent', 2010, 0.8),
        ('modern', 2000, 1.0),
        ('modern', 1985, 0.8),
        ('golden_age', 1970, 1.0),
        ('golden_age', 1990, 0.6),
        ('any', 1950, 1.0),
    ]

    print("Testing era scoring (movie_year vs selected_era):\n")
    for era_id, movie_year, expected_score in test_cases:
        score = TimePeriodFilter.calculate_era_score(movie_year, era_id)
        era_label = TimePeriodFilter.get_era_short_label(era_id)
        print(f"  {movie_year} vs {era_label}: {score:.2f} (expected ~{expected_score:.2f})")

    # Test movie filtering
    print(f"\n{'='*70}")
    print("MOVIE FILTERING TEST")
    print(f"{'='*70}\n")

    test_movies = [
        {'title': 'The Matrix', 'year': 1999},
        {'title': 'Inception', 'year': 2010},
        {'title': 'Dune', 'year': 2021},
        {'title': 'Godfather', 'year': 1972},
        {'title': 'Casablanca', 'year': 1942},
    ]

    selected_era = 'modern'
    filtered = TimePeriodFilter.filter_movies_by_era(
        test_movies.copy(), selected_era, strict=False
    )

    print(f"Movies scored for era '{selected_era}':\n")
    for movie in sorted(filtered, key=lambda m: m.get('era_score', 0), reverse=True):
        score = movie.get('era_score', 1.0)
        print(f"  {movie['title']} ({movie['year']}): {score:.2f}")
