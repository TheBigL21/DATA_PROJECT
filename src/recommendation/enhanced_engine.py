"""
ENHANCED RECOMMENDATION ENGINE

Extends the base recommendation engine with 6-question filtering system.
Supports: evening type, multi-genre, age, runtime, quality, and popularity filters.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommendation.recommendation_engine import RecommendationEngine, load_system as load_base_system
from recommendation.region_weighting import RegionWeighting
from models.content_similarity import ContentSimilarity


class EnhancedRecommendationEngine:
    """Enhanced engine with 6-question filter support + content similarity"""

    def __init__(self, base_engine: RecommendationEngine, content_similarity: Optional[ContentSimilarity] = None):
        """
        Args:
            base_engine: Base recommendation engine instance
            content_similarity: Optional content similarity model for semantic matching
        """
        self.base_engine = base_engine
        self.movies = base_engine.movies
        self.region_weighter = RegionWeighting()
        self.content_similarity = content_similarity

    def recommend(
        self,
        user_id: int,
        evening_type: str,
        selected_genres: List[str],  # 1-2 genres
        age_preference: str,
        runtime_pref: str,
        quality_level: str,
        popularity_level: str,
        session_history: List[Dict],
        user_region: str = "Other",  # NEW: User's region
        selected_keywords: List[str] = None,  # NEW: Selected keywords
        top_k: int = 20
    ) -> List[int]:
        """
        Generate recommendations with enhanced 6-question filtering

        Args:
            user_id: User ID
            evening_type: Chill/Date/Family/Friends night
            selected_genres: List of 1-2 genres
            age_preference: New Gen/Grown Up/Old School/Real Oldtimer/Doesn't Matter
            runtime_pref: Quick/Standard/Epic/Marathon/Doesn't matter
            quality_level: Only best/Good/Hidden gems/Try anything
            popularity_level: Blockbusters/Popular/Hidden gems/Mix
            session_history: Previous interactions
            user_region: User's region (for regional weighting)
            top_k: Number of recommendations

        Returns:
            List of movie IDs
        """
        # STAGE 1: Apply all filters (including keywords)
        filtered_movies = self._apply_filters(
            selected_genres,
            age_preference,
            runtime_pref,
            quality_level,
            popularity_level,
            user_region,  # Pass region for old movie limiting
            selected_keywords  # NEW: Pass keywords
        )

        if len(filtered_movies) < top_k:
            # Relax filters if too restrictive
            filtered_movies = self._relax_filters(
                selected_genres,
                age_preference,
                runtime_pref,
                quality_level,
                popularity_level,
                user_region,
                min_required=top_k
            )

        # STAGE 2: Apply region-based weighting
        filtered_movies['base_score'] = filtered_movies['avg_rating'] * np.log1p(filtered_movies['num_votes'])

        if user_region and user_region != 'Other':
            filtered_movies = self.region_weighter.apply_region_weighting(
                filtered_movies,
                user_region,
                score_column='base_score'
            )
            score_col = 'weighted_score'
        else:
            score_col = 'base_score'

        # STAGE 3: Use base engine for scoring
        # Create a temporary filtered movie set for base engine
        filtered_ids = set(filtered_movies['movie_id'])

        # Get base recommendations (using first genre for compatibility)
        primary_genre = selected_genres[0]
        mood_decade = self._age_to_decade(age_preference)

        base_recommendations = self.base_engine.recommend(
            user_id=user_id,
            mood_genre=primary_genre,
            mood_decade=mood_decade,
            session_history=session_history,
            top_k=top_k * 3  # Get more candidates
        )

        # Filter base recommendations to only include our filtered set
        final_recommendations = [
            movie_id for movie_id in base_recommendations
            if movie_id in filtered_ids
        ]

        # If not enough, add from filtered_movies directly (sorted by weighted score)
        if len(final_recommendations) < top_k:
            filtered_sorted = filtered_movies.sort_values(score_col, ascending=False)
            additional = filtered_sorted[
                ~filtered_sorted['movie_id'].isin(final_recommendations)
            ]['movie_id'].tolist()
            final_recommendations.extend(additional[:top_k - len(final_recommendations)])

        # STAGE 4: Limit very old movies when "Doesn't Matter" era is selected
        if age_preference == "Doesn't Matter":
            final_recommendations = self._limit_old_movies(
                final_recommendations,
                filtered_movies,
                max_old_movies=2,
                top_k=top_k
            )

        return final_recommendations[:top_k]

    def _apply_filters(
        self,
        selected_genres: List[str],
        age_preference: str,
        runtime_pref: str,
        quality_level: str,
        popularity_level: str,
        user_region: str = "Other",
        selected_keywords: List[str] = None
    ) -> pd.DataFrame:
        """Apply all filters including keywords"""

        filtered = self.movies.copy()

        # 0. Keyword filter (if provided)
        if selected_keywords and len(selected_keywords) > 0:
            # Filter movies that have at least one of the selected keywords
            def has_keyword(movie_keywords):
                if movie_keywords is None or (isinstance(movie_keywords, float) and pd.isna(movie_keywords)):
                    return False
                # Convert to list if numpy array
                kw_list = list(movie_keywords) if isinstance(movie_keywords, (list, tuple, np.ndarray)) else []
                return any(kw in kw_list for kw in selected_keywords)

            filtered = filtered[filtered['keywords'].apply(has_keyword)]

        # 1. Genre filter (OR logic for multi-genre)
        if selected_genres and selected_genres[0] != "Doesn't Matter":
            genre_mask = filtered['genres'].apply(
                lambda x: any(g in x for g in selected_genres)
            )
            filtered = filtered[genre_mask]

        # 2. Age filter with OLD MOVIE LIMITATION
        if age_preference != "Doesn't Matter":
            year_min, year_max = self._age_to_year_range(age_preference)
            filtered = filtered[
                (filtered['year'] >= year_min) &
                (filtered['year'] <= year_max)
            ]
        else:
            # When "Doesn't Matter", apply smart old movie limiting
            # Strategy: Separate very old movies (>50 years) and newer movies
            current_year = datetime.now().year
            very_old_cutoff = current_year - 50  # Movies from 1975 or earlier

            very_old = filtered[filtered['year'] <= very_old_cutoff].copy()
            newer = filtered[filtered['year'] > very_old_cutoff].copy()

            # Tag movies for later limiting
            very_old['is_very_old'] = True
            newer['is_very_old'] = False

            filtered = pd.concat([very_old, newer], ignore_index=True)

        # 3. Runtime filter
        if runtime_pref != "Doesn't matter":
            runtime_min, runtime_max = self._runtime_to_range(runtime_pref)
            filtered = filtered[
                (filtered['runtime'] >= runtime_min) &
                (filtered['runtime'] <= runtime_max)
            ]

        # 4. Quality filter (using new rating_100 scale!)
        if quality_level != "Open to Anything (All)":
            rating_min = self._quality_to_rating(quality_level)
            # Use rating_100 if available, fallback to composite_rating * 10
            if 'rating_100' in filtered.columns:
                rating_col = 'rating_100'
            elif 'composite_rating' in filtered.columns:
                filtered['rating_100'] = filtered['composite_rating'] * 10
                rating_col = 'rating_100'
            else:
                filtered['rating_100'] = filtered['avg_rating'] * 10
                rating_col = 'rating_100'

            filtered = filtered[filtered[rating_col] >= rating_min]

        # 5. Popularity filter
        if popularity_level != "Mix of everything":
            votes_min, votes_max = self._popularity_to_votes(popularity_level)
            if votes_max:
                filtered = filtered[
                    (filtered['num_votes'] >= votes_min) &
                    (filtered['num_votes'] < votes_max)
                ]
            else:
                filtered = filtered[filtered['num_votes'] >= votes_min]

        return filtered

    def _limit_old_movies(
        self,
        recommendations: List[int],
        filtered_movies: pd.DataFrame,
        max_old_movies: int = 2,
        top_k: int = 10
    ) -> List[int]:
        """
        Limit very old movies (>50 years) in recommendations

        Args:
            recommendations: List of movie IDs
            filtered_movies: DataFrame with movie info and 'is_very_old' column
            max_old_movies: Maximum number of very old movies to include
            top_k: Target number of recommendations

        Returns:
            Adjusted list of movie IDs with limited old movies
        """
        if 'is_very_old' not in filtered_movies.columns:
            return recommendations

        # Create movie lookup
        movie_lookup = filtered_movies.set_index('movie_id')['is_very_old'].to_dict()

        # Separate old and newer movies
        old_movies = []
        newer_movies = []

        for movie_id in recommendations:
            if movie_id in movie_lookup and movie_lookup[movie_id]:
                old_movies.append(movie_id)
            else:
                newer_movies.append(movie_id)

        # Limit old movies
        limited_old = old_movies[:max_old_movies]

        # Combine: prefer newer movies, add limited old movies at the end
        final = newer_movies + limited_old

        # If we need more movies, add from remaining old movies
        if len(final) < top_k:
            remaining_old = old_movies[max_old_movies:]
            final.extend(remaining_old[:top_k - len(final)])

        return final[:top_k]

    def _relax_filters(
        self,
        selected_genres: List[str],
        age_preference: str,
        runtime_pref: str,
        quality_level: str,
        popularity_level: str,
        user_region: str,
        min_required: int
    ) -> pd.DataFrame:
        """Progressively relax filters if too restrictive"""

        # Try relaxing in order of priority
        relaxation_steps = [
            (quality_level, "Good movies"),  # Relax quality first
            (popularity_level, "Mix of everything"),  # Then popularity
            (runtime_pref, "Doesn't matter"),  # Then runtime
            (age_preference, "Doesn't Matter"),  # Age last
        ]

        current_quality = quality_level
        current_popularity = popularity_level
        current_runtime = runtime_pref
        current_age = age_preference

        for _ in range(len(relaxation_steps)):
            filtered = self._apply_filters(
                selected_genres,
                current_age,
                current_runtime,
                current_quality,
                current_popularity,
                user_region
            )

            if len(filtered) >= min_required:
                return filtered

            # Relax next filter
            if current_quality != "I'll try anything":
                current_quality = "I'll try anything"
            elif current_popularity != "Mix of everything":
                current_popularity = "Mix of everything"
            elif current_runtime != "Doesn't matter":
                current_runtime = "Doesn't matter"
            elif current_age != "Doesn't Matter":
                current_age = "Doesn't Matter"

        # Final fallback: just genre
        return self._apply_filters(
            selected_genres,
            "Doesn't Matter",
            "Doesn't matter",
            "I'll try anything",
            "Mix of everything",
            user_region
        )

    def _age_to_year_range(self, age_preference: str) -> tuple:
        """Convert age preference to year range"""
        current_year = 2025

        ranges = {
            "Less than 5 years old": (current_year - 5, current_year),
            "Less than 10 years old": (current_year - 10, current_year),
            "Less than 20 years old": (current_year - 20, current_year),
            # Legacy support for old labels
            "New Gen (Last 5 years)": (current_year - 5, current_year),
            "Grown Up (10-25 years ago)": (current_year - 25, current_year - 10),
            "Old School (25-50 years ago)": (current_year - 50, current_year - 25),
            "Real Oldtimer (50+ years ago)": (1900, current_year - 50)
        }

        return ranges.get(age_preference, (1900, current_year))

    def _age_to_decade(self, age_preference: str) -> str:
        """Convert age preference to decade string for base engine"""
        mappings = {
            "Less than 5 years old": "2020s",
            "Less than 10 years old": "2020s",
            "Less than 20 years old": "2010s",
            # Legacy support for old labels
            "New Gen (Last 5 years)": "2020s",
            "Grown Up (10-25 years ago)": "2000s",
            "Old School (25-50 years ago)": "1990s",
            "Real Oldtimer (50+ years ago)": "1970s"
        }
        return mappings.get(age_preference, "2000s")

    def _runtime_to_range(self, runtime_pref: str) -> tuple:
        """Convert runtime preference to minute range"""
        ranges = {
            "Quick watch (<90 min)": (0, 90),
            "Standard movie (90-120 min)": (90, 120),
            "Epic experience (120-180 min)": (120, 180),
            "Marathon (180+ min)": (180, 999)
        }
        return ranges.get(runtime_pref, (0, 999))

    def _quality_to_rating(self, quality_level: str) -> float:
        """Convert quality level to minimum rating (0-100 scale)"""
        ratings = {
            # New 0-100 scale based on statistical distribution
            "Real Masterpieces (78+)": 78.0,
            "Very Good Movies (71+)": 71.0,
            "Good Movies (66+)": 66.0,
            "Hidden Gems (61+)": 61.0,
            "Open to Anything (All)": 0.0,
            # Old labels for backward compatibility (converted to 100 scale)
            "Masterpieces only": 80.0,
            "Highly rated films": 70.0,
            "Good movies & hidden gems": 60.0,
            "I'm open to anything": 50.0,
            "Only the best (8.0+)": 80.0,
            "Good movies (7.0+)": 70.0,
            "Hidden gems (6.0+)": 60.0,
            "I'll try anything (5.0+)": 50.0
        }
        return ratings.get(quality_level, 50.0)

    def _popularity_to_votes(self, popularity_level: str) -> tuple:
        """Convert popularity level to vote range"""
        ranges = {
            "Blockbusters (100K+ votes)": (100000, None),
            "Popular picks (10K-100K votes)": (10000, 100000),
            "Hidden gems (1K-10K votes)": (1000, 10000)
        }
        return ranges.get(popularity_level, (1000, None))


def load_system(models_dir: Path, movies_path: Path) -> EnhancedRecommendationEngine:
    """
    Load enhanced recommendation system

    Args:
        models_dir: Directory with trained models
        movies_path: Path to movies.parquet

    Returns:
        EnhancedRecommendationEngine instance
    """
    print("Loading base recommendation system...")
    base_engine = load_base_system(models_dir, movies_path)

    # Try to load content similarity model (optional)
    content_sim = None
    try:
        content_sim_path = models_dir / 'content_similarity.pkl'
        if content_sim_path.exists():
            print("Loading content similarity model...")
            content_sim = ContentSimilarity.load(models_dir)
        else:
            print("Content similarity model not found (optional) - skipping")
    except Exception as e:
        print(f"Could not load content similarity: {e}")

    print("Initializing enhanced recommendation engine...")
    enhanced_engine = EnhancedRecommendationEngine(base_engine, content_sim)

    print("Enhanced system ready.")
    return enhanced_engine
