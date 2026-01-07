"""
User Preference Inference Engine

Infers user preferences from keyword selections and session behavior
without asking explicit questions.

Inferences:
1. Quality expectations (from keyword sophistication)
2. Popularity preference (from session voting patterns)
3. Pacing preference (from keywords + session patterns)
"""

import numpy as np
from typing import Dict, List, Optional, Any


def infer_user_profile(selected_keywords: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer user quality expectations from keyword selections.

    Logic:
    - Selects classic era keywords → cinephile, higher quality bar
    - Selects book adaptation → prestige focus, higher quality bar
    - Selects 3+ specific keywords → explorer, lower quality bar for variety
    - Selects few/no keywords → casual, wants safe popular picks

    Args:
        selected_keywords: Dict with keys 'age', 'general', 'adaptation', 'pacing'

    Returns:
        Dict with 'quality_floor', 'quality_weight', 'mode'
    """
    age_keyword = selected_keywords.get('age')
    general_keywords = selected_keywords.get('general', [])
    wants_adaptation = selected_keywords.get('adaptation', False)

    # Cinephile mode: classic era keywords
    if age_keyword in ['90s_classics', '80s_classics', 'golden_age']:
        return {
            'quality_floor': 7.5,
            'quality_weight': 0.20,
            'mode': 'cinephile'
        }

    # Quality-focused mode: book adaptations
    if wants_adaptation:
        return {
            'quality_floor': 7.0,
            'quality_weight': 0.18,
            'mode': 'quality_focused'
        }

    # Explorer mode: many specific keywords
    if len(general_keywords) >= 3:
        return {
            'quality_floor': 6.5,
            'quality_weight': 0.12,
            'mode': 'explorer'
        }

    # Safe picks mode: few keywords selected
    if len(general_keywords) <= 1 and not age_keyword:
        return {
            'quality_floor': 7.0,
            'quality_weight': 0.18,
            'mode': 'safe_picks'
        }

    # Casual mode: recent films
    if age_keyword in ['recent', '2010s', '2000s']:
        return {
            'quality_floor': 6.5,
            'quality_weight': 0.12,
            'mode': 'casual'
        }

    # Default: balanced
    return {
        'quality_floor': 6.5,
        'quality_weight': 0.15,
        'mode': 'balanced'
    }


def infer_popularity_preference(session_history: List[dict]) -> str:
    """
    Infer popularity preference from session voting patterns.

    Analyzes vote counts of movies user swiped right on:
    - High avg votes (>200K) → user likes popular films
    - Low avg votes (<30K) → user likes hidden gems
    - Medium → balanced

    Args:
        session_history: List of dicts with 'action' and 'num_votes' fields

    Returns:
        'popular', 'hidden', or 'balanced'
    """
    if not session_history or len(session_history) < 3:
        return 'balanced'

    liked_movies = [m for m in session_history if m.get('action') == 'up']

    if len(liked_movies) < 3:
        return 'balanced'

    # Calculate average vote count
    vote_counts = [m.get('num_votes', 0) for m in liked_movies]
    avg_votes = np.mean(vote_counts)

    if avg_votes > 200000:
        return 'popular'
    elif avg_votes < 30000:
        return 'hidden'
    else:
        return 'balanced'


def calculate_popularity_boost(movie_votes: int, popularity_mode: str) -> float:
    """
    Calculate popularity boost multiplier for a movie.

    Args:
        movie_votes: Number of votes the movie has
        popularity_mode: 'popular', 'hidden', or 'balanced'

    Returns:
        Multiplier from 0.8 to 1.3
    """
    if popularity_mode == 'popular':
        # Boost high-vote films (200K+ votes get up to 1.3x)
        boost = min(1.3, 1.0 + (movie_votes / 1000000))
        return boost

    elif popularity_mode == 'hidden':
        # Boost low-vote films (under 50K get up to 1.2x)
        if movie_votes < 50000:
            boost = min(1.2, 1.0 + ((100000 - movie_votes) / 500000))
        else:
            boost = 1.0
        return max(0.8, boost)

    else:  # balanced
        return 1.0


def infer_pacing_from_keywords(selected_keywords: Dict[str, Any]) -> Optional[str]:
    """
    Extract pacing preference from keyword selections.

    Args:
        selected_keywords: Dict with 'pacing' key

    Returns:
        'fast', 'slow', or None
    """
    pacing_keywords = selected_keywords.get('pacing', [])

    if not pacing_keywords:
        return None

    if isinstance(pacing_keywords, str):
        pacing_keywords = [pacing_keywords]

    if 'fast_paced' in pacing_keywords:
        return 'fast'
    elif 'slow_burn' in pacing_keywords:
        return 'slow'

    return None


def infer_pacing_from_session(session_history: List[dict]) -> Optional[str]:
    """
    Infer pacing preference from session behavior.

    Args:
        session_history: List of dicts with 'action' and 'pacing_score' fields

    Returns:
        'fast', 'slow', or None
    """
    if not session_history or len(session_history) < 3:
        return None

    liked_movies = [m for m in session_history if m.get('action') == 'up']

    if len(liked_movies) < 3:
        return None

    # Calculate average pacing
    pacing_scores = [m.get('pacing_score', 0.5) for m in liked_movies]
    avg_pacing = np.mean(pacing_scores)

    if avg_pacing > 0.7:
        return 'fast'
    elif avg_pacing < 0.4:
        return 'slow'
    else:
        return None


def build_user_model(selected_keywords: Dict[str, Any],
                     session_history: List[dict]) -> Dict[str, Any]:
    """
    Build complete user preference model from keywords and session.

    Args:
        selected_keywords: User's keyword selections
        session_history: User's session voting history

    Returns:
        Dict containing:
        - quality: quality profile dict
        - popularity: popularity mode string
        - pacing: pacing preference string or None
    """
    # Infer quality expectations from keywords
    quality_profile = infer_user_profile(selected_keywords)

    # Infer popularity preference from session (if enough data)
    popularity_mode = infer_popularity_preference(session_history)

    # Infer pacing preference
    # First try keywords (explicit), then session (implicit)
    pacing_pref = infer_pacing_from_keywords(selected_keywords)
    if not pacing_pref and len(session_history) >= 3:
        pacing_pref = infer_pacing_from_session(session_history)

    return {
        'quality': quality_profile,
        'popularity': popularity_mode,
        'pacing': pacing_pref
    }


if __name__ == '__main__':
    # Test inference engine
    print("Testing Inference Engine\n")
    print("=" * 60)

    test_cases = [
        {
            'name': 'Cinephile (90s classics)',
            'keywords': {'age': '90s_classics', 'general': ['neo noir'], 'adaptation': False},
            'session': []
        },
        {
            'name': 'Explorer (3+ keywords)',
            'keywords': {'age': 'recent', 'general': ['heist', 'double cross', 'undercover'], 'adaptation': False},
            'session': []
        },
        {
            'name': 'Casual (recent, few keywords)',
            'keywords': {'age': 'recent', 'general': [], 'adaptation': False},
            'session': []
        },
        {
            'name': 'Book lover',
            'keywords': {'age': None, 'general': [], 'adaptation': True},
            'session': []
        },
        {
            'name': 'Popularity: Popular (after session)',
            'keywords': {'age': 'recent', 'general': ['heist'], 'adaptation': False},
            'session': [
                {'action': 'up', 'num_votes': 500000, 'pacing_score': 0.8},
                {'action': 'up', 'num_votes': 800000, 'pacing_score': 0.75},
                {'action': 'up', 'num_votes': 1200000, 'pacing_score': 0.85},
            ]
        },
        {
            'name': 'Popularity: Hidden (after session)',
            'keywords': {'age': '90s_classics', 'general': ['neo noir'], 'adaptation': False},
            'session': [
                {'action': 'up', 'num_votes': 15000, 'pacing_score': 0.3},
                {'action': 'up', 'num_votes': 25000, 'pacing_score': 0.35},
                {'action': 'up', 'num_votes': 18000, 'pacing_score': 0.25},
            ]
        },
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  Keywords: {test['keywords']}")

        model = build_user_model(test['keywords'], test['session'])

        print(f"  → Quality mode: {model['quality']['mode']}")
        print(f"  → Quality floor: {model['quality']['quality_floor']}")
        print(f"  → Quality weight: {model['quality']['quality_weight']}")
        print(f"  → Popularity: {model['popularity']}")
        print(f"  → Pacing: {model['pacing']}")

        if model['popularity'] == 'popular':
            print(f"  → Boost for 500K votes: {calculate_popularity_boost(500000, 'popular'):.2f}x")
        elif model['popularity'] == 'hidden':
            print(f"  → Boost for 20K votes: {calculate_popularity_boost(20000, 'hidden'):.2f}x")

    print("\n" + "=" * 60)
