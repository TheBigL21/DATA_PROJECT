"""
Age-based Keyword System

Provides age/era keywords as part of the keyword selection interface.
Each age keyword defines a time period filter with decay scoring.
"""

from typing import Dict, List, Callable


AGE_KEYWORDS: Dict[str, dict] = {
    "recent": {
        "label": "Recent (2020-2025)",
        "filter": lambda year: year >= 2020,
        "target_year": 2023,
        "decay_rate": 0.15,  # 15% score reduction per year away from target
        "priority": 1
    },
    "2010s": {
        "label": "2010s Era",
        "filter": lambda year: 2010 <= year < 2020,
        "target_year": 2015,
        "decay_rate": 0.10,
        "priority": 2
    },
    "2000s": {
        "label": "2000s Films",
        "filter": lambda year: 2000 <= year < 2010,
        "target_year": 2005,
        "decay_rate": 0.08,
        "priority": 3
    },
    "90s_classics": {
        "label": "90s Classics",
        "filter": lambda year: 1990 <= year < 2000,
        "target_year": 1995,
        "decay_rate": 0.05,
        "priority": 4
    },
    "80s_classics": {
        "label": "80s Classics",
        "filter": lambda year: 1980 <= year < 1990,
        "target_year": 1985,
        "decay_rate": 0.05,
        "priority": 5
    },
    "golden_age": {
        "label": "Golden Age (1950-1979)",
        "filter": lambda year: 1950 <= year < 1980,
        "target_year": 1970,
        "decay_rate": 0.03,
        "priority": 6
    },
}


def get_age_keywords_for_display() -> List[str]:
    """
    Return age keywords to display to user.

    Returns 2 age keywords: one recent, one classic era.

    Returns:
        List of age keyword IDs (e.g., ['recent', '90s_classics'])
    """
    # Default: show recent + 90s classics
    # This covers modern viewers and cinephiles
    return ["recent", "90s_classics"]


def calculate_age_score(movie_year: int, age_keyword: str) -> float:
    """
    Calculate age match score for a movie based on selected age keyword.

    Uses decay function instead of hard cutoff:
    - Movies within target range get 1.0
    - Movies outside range decay based on distance from target year

    Args:
        movie_year: Year the movie was released
        age_keyword: Selected age keyword ID (e.g., 'recent')

    Returns:
        Score from 0.1 to 1.0 indicating age match quality
    """
    if age_keyword not in AGE_KEYWORDS:
        return 1.0  # No penalty if invalid keyword

    config = AGE_KEYWORDS[age_keyword]
    target_year = config['target_year']
    decay_rate = config['decay_rate']

    # Check if within preferred range
    if config['filter'](movie_year):
        # Perfect match - within target era
        return 1.0

    # Outside range - apply decay based on distance
    years_away = abs(movie_year - target_year)
    score = max(0.1, 1.0 - (years_away * decay_rate))

    return score


def get_age_keyword_display_info() -> List[Dict[str, str]]:
    """
    Get display information for age keywords in UI.

    Returns:
        List of dicts with 'id' and 'label' for each displayed age keyword
    """
    display_keywords = get_age_keywords_for_display()

    return [
        {
            'id': keyword_id,
            'label': AGE_KEYWORDS[keyword_id]['label']
        }
        for keyword_id in display_keywords
    ]


if __name__ == '__main__':
    # Test age scoring
    print("Testing age score calculations:\n")

    test_cases = [
        ('recent', 2024, "Brand new film"),
        ('recent', 2020, "Start of range"),
        ('recent', 2015, "5 years before target"),
        ('recent', 2010, "13 years before target"),
        ('90s_classics', 1995, "Perfect match"),
        ('90s_classics', 1998, "Near target"),
        ('90s_classics', 2005, "10 years after target"),
        ('90s_classics', 1985, "10 years before target"),
    ]

    for keyword, year, description in test_cases:
        score = calculate_age_score(year, keyword)
        print(f"{keyword:15} | {year} | {score:.2f} | {description}")
