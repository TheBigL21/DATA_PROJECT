"""
SYNTHETIC INTERACTION GENERATOR

Purpose: Generate realistic user interaction data for model training.

Since no real users exist yet, we simulate user behavior based on:
- Genre preferences (users have favorite genres)
- Year preferences (users prefer certain decades)
- Rating sensitivity (some users prefer highly-rated movies)
- Session coherence (within a session, users explore similar movies)

Output: interactions.parquet with schema:
- interaction_id: int
- user_id: int
- movie_id: int
- session_id: str (user_id + session_number)
- timestamp: datetime
- action: str ('left', 'right', 'up')
- action_value: float (0.0, 0.3, 1.0)
- position_in_session: int (0, 1, 2, ...)
- mood_genre: str (primary genre user selected at session start)
- mood_decade: str ('1990s', '2000s', etc.)

Generation Logic:
1. Create N synthetic users with preferences
2. For each user, generate M sessions
3. For each session:
   - Pick mood (genre + decade)
   - Simulate swipe sequence until 'up' action or max swipes
   - Earlier swipes more exploratory, later swipes converge to preferences
   - Generate realistic action probabilities based on movie-preference match
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_synthetic_users(num_users: int, all_genres: list) -> pd.DataFrame:
    """
    Generate synthetic users with genre and decade preferences.

    Each user has:
    - Preferred genres (2-4 genres with weights)
    - Preferred decades (normal distribution around a mean decade)
    - Rating threshold (minimum rating they typically accept)
    - Exploration rate (how often they try movies outside preferences)

    Returns DataFrame: user_id, genre_prefs (dict), decade_mean, decade_std, rating_threshold, exploration_rate
    """
    users = []

    for user_id in range(num_users):
        # Pick 2-4 preferred genres
        num_genres = np.random.randint(2, 5)
        preferred_genres = np.random.choice(all_genres, size=num_genres, replace=False)
        # Assign weights (sums to 1)
        weights = np.random.dirichlet(np.ones(num_genres))
        genre_prefs = dict(zip(preferred_genres, weights))

        # Decade preference: centered around random year between 1980-2015
        decade_mean = np.random.randint(1980, 2016)
        decade_std = np.random.uniform(10, 20)  # Some users flexible, some strict

        # Rating threshold: most users prefer 6.5+
        rating_threshold = np.random.uniform(6.0, 8.0)

        # Exploration rate: how often user swipes right/up on non-preferred movie
        exploration_rate = np.random.uniform(0.05, 0.25)

        users.append({
            'user_id': user_id,
            'genre_prefs': genre_prefs,
            'decade_mean': decade_mean,
            'decade_std': decade_std,
            'rating_threshold': rating_threshold,
            'exploration_rate': exploration_rate
        })

    return pd.DataFrame(users)


def compute_preference_score(movie: dict, user: dict) -> float:
    """
    Compute how well a movie matches user preferences.

    Factors:
    - Genre match: weighted overlap between movie genres and user preferred genres
    - Year match: Gaussian probability based on user decade preference
    - Rating match: binary indicator if rating above threshold

    Returns score in [0, 1]
    """
    # Genre score
    genre_score = 0.0
    movie_genres = movie['genres']
    user_genre_prefs = user['genre_prefs']

    for genre in movie_genres:
        if genre in user_genre_prefs:
            genre_score += user_genre_prefs[genre]

    # Normalize by number of movie genres to avoid bias toward multi-genre movies
    if len(movie_genres) > 0:
        genre_score /= len(movie_genres)

    # Year score: Gaussian centered at user decade_mean
    year_diff = abs(movie['year'] - user['decade_mean'])
    year_score = np.exp(-0.5 * (year_diff / user['decade_std']) ** 2)

    # Rating score: binary
    rating_score = 1.0 if movie['avg_rating'] >= user['rating_threshold'] else 0.3

    # Weighted combination
    preference_score = 0.5 * genre_score + 0.3 * year_score + 0.2 * rating_score

    return np.clip(preference_score, 0, 1)


def simulate_session_actions(
    user: dict,
    movies_pool: pd.DataFrame,
    session_id: str,
    mood_genre: str,
    mood_decade: str,
    max_swipes: int = 30
) -> list:
    """
    Simulate one user session: sequence of swipes until 'up' or max_swipes reached.

    Logic:
    - Start with movies matching mood (genre + decade filter)
    - For each movie shown:
      - Compute preference score
      - Sample action based on score:
        * High score -> likely 'up' or 'right'
        * Low score -> likely 'left'
      - Add noise (exploration rate)
    - Continue until 'up' action or max_swipes

    Returns list of interaction dicts
    """
    # Filter movies to mood
    # Mood genre filter
    mood_movies = movies_pool[
        movies_pool['genres'].apply(lambda g: mood_genre in g)
    ].copy()

    # Mood decade filter: within ±15 years of decade midpoint
    decade_year = int(mood_decade[:4])  # e.g., '1990s' -> 1990
    mood_movies = mood_movies[
        (mood_movies['year'] >= decade_year - 15) &
        (mood_movies['year'] <= decade_year + 15)
    ]

    if len(mood_movies) == 0:
        # Fallback: use all movies if mood filter too restrictive
        mood_movies = movies_pool.copy()

    # Shuffle and sample subset
    mood_movies = mood_movies.sample(n=min(len(mood_movies), max_swipes * 2), replace=False)

    interactions = []
    position = 0
    base_timestamp = datetime.now() - timedelta(days=np.random.randint(1, 365))

    for idx, movie in mood_movies.iterrows():
        if position >= max_swipes:
            break

        # Compute preference score
        movie_dict = movie.to_dict()
        pref_score = compute_preference_score(movie_dict, user)

        # Add exploration noise
        if np.random.random() < user['exploration_rate']:
            pref_score += np.random.uniform(0, 0.3)
        pref_score = np.clip(pref_score, 0, 1)

        # Sample action based on preference score
        # High score -> more likely up/right
        # Low score -> more likely left
        rand = np.random.random()

        if pref_score > 0.75:
            # Strong match: 60% up, 30% right, 10% left
            if rand < 0.6:
                action = 'up'
            elif rand < 0.9:
                action = 'right'
            else:
                action = 'left'
        elif pref_score > 0.5:
            # Moderate match: 20% up, 50% right, 30% left
            if rand < 0.2:
                action = 'up'
            elif rand < 0.7:
                action = 'right'
            else:
                action = 'left'
        elif pref_score > 0.3:
            # Weak match: 5% up, 25% right, 70% left
            if rand < 0.05:
                action = 'up'
            elif rand < 0.3:
                action = 'right'
            else:
                action = 'left'
        else:
            # Very weak: 90% left
            if rand < 0.9:
                action = 'left'
            else:
                action = 'right'

        # Map action to value
        action_value = {'left': 0.0, 'right': 0.3, 'up': 1.0}[action]

        # Record interaction
        interactions.append({
            'user_id': user['user_id'],
            'movie_id': movie['movie_id'],
            'session_id': session_id,
            'timestamp': base_timestamp + timedelta(seconds=position * 10),
            'action': action,
            'action_value': action_value,
            'position_in_session': position,
            'mood_genre': mood_genre,
            'mood_decade': mood_decade
        })

        position += 1

        # Stop if user selected movie
        if action == 'up':
            break

    return interactions


def generate_interactions(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    sessions_per_user: int = 10,
    max_swipes_per_session: int = 30
) -> pd.DataFrame:
    """
    Generate all synthetic interactions for all users.

    For each user:
    - Generate multiple sessions
    - Each session has random mood (genre + decade)
    - Simulate swipe sequence

    Returns DataFrame with all interactions
    """
    all_interactions = []
    all_genres = sorted(set([g for genres in movies['genres'] for g in genres]))
    decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

    interaction_id = 0

    for _, user in users.iterrows():
        user_dict = user.to_dict()

        for session_num in range(sessions_per_user):
            # Pick random mood
            mood_genre = np.random.choice(list(user_dict['genre_prefs'].keys()))
            mood_decade = np.random.choice(decades)

            session_id = f"user{user_dict['user_id']}_session{session_num}"

            # Simulate session
            session_interactions = simulate_session_actions(
                user_dict,
                movies,
                session_id,
                mood_genre,
                mood_decade,
                max_swipes_per_session
            )

            # Add interaction IDs
            for inter in session_interactions:
                inter['interaction_id'] = interaction_id
                interaction_id += 1
                all_interactions.append(inter)

        if (user_dict['user_id'] + 1) % 100 == 0:
            print(f"Generated interactions for {user_dict['user_id'] + 1} users...")

    return pd.DataFrame(all_interactions)


def main(movies_path: Path, output_dir: Path, num_users: int = 1000, sessions_per_user: int = 10):
    """
    Main execution: Generate synthetic interaction dataset.

    Args:
        movies_path: Path to movies.parquet
        output_dir: Directory to save interactions.parquet
        num_users: Number of synthetic users to create
        sessions_per_user: Number of sessions per user
    """
    print(f"Loading movies from {movies_path}...")
    movies = pd.read_parquet(movies_path)
    print(f"Loaded {len(movies)} movies")

    # Get all unique genres
    all_genres = sorted(set([g for genres in movies['genres'] for g in genres]))
    print(f"Found {len(all_genres)} unique genres: {all_genres[:10]}...")

    print(f"\nGenerating {num_users} synthetic users...")
    users = create_synthetic_users(num_users, all_genres)
    print(f"Created {len(users)} users")

    print(f"\nGenerating interactions ({sessions_per_user} sessions per user)...")
    interactions = generate_interactions(users, movies, sessions_per_user)

    print(f"\nGenerated {len(interactions)} total interactions")
    print(f"Action distribution:")
    print(interactions['action'].value_counts())
    print(f"\nAverage swipes per session: {interactions.groupby('session_id').size().mean():.1f}")

    # Save
    interactions_path = output_dir / 'interactions.parquet'
    interactions.to_parquet(interactions_path, index=False)
    print(f"\nSaved interactions to {interactions_path}")

    # Also save users for reference
    users_path = output_dir / 'synthetic_users.parquet'
    users.to_parquet(users_path, index=False)
    print(f"Saved users to {users_path}")

    return interactions


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate_synthetic_interactions.py <movies_parquet> <output_dir> [num_users] [sessions_per_user]")
        print("Example: python generate_synthetic_interactions.py ../../output/processed/movies.parquet ../../output/processed 1000 10")
        sys.exit(1)

    movies_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    num_users = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    sessions_per_user = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    output_dir.mkdir(parents=True, exist_ok=True)

    main(movies_path, output_dir, num_users, sessions_per_user)
