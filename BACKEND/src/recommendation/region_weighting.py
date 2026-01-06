"""
REGION-BASED MOVIE WEIGHTING

Applies region-specific popularity boosts to movies based on user's location.
Prioritizes local cinema and Hollywood movies for certain regions.
"""

import pandas as pd
from typing import List, Dict


class RegionWeighting:
    """Apply region-based weighting to movie recommendations"""

    def __init__(self):
        """
        Initialize region weighting system

        Region categories:
        - North America: Boost US/Canada + Hollywood movies
        - Europe: Boost European + Hollywood movies
        - Asia: Boost Asian + popular international cinema
        - Latin America: Boost Latin American + Hollywood movies
        - Other: Balanced approach
        """

        # Define country-to-region mappings for movies
        self.country_to_region = {
            # North America
            'USA': 'North America',
            'Canada': 'North America',
            'United States': 'North America',

            # Europe
            'UK': 'Europe',
            'United Kingdom': 'Europe',
            'France': 'Europe',
            'Germany': 'Europe',
            'Italy': 'Europe',
            'Spain': 'Europe',
            'Netherlands': 'Europe',
            'Sweden': 'Europe',
            'Denmark': 'Europe',
            'Norway': 'Europe',
            'Poland': 'Europe',
            'Belgium': 'Europe',
            'Austria': 'Europe',
            'Switzerland': 'Europe',
            'Ireland': 'Europe',
            'Portugal': 'Europe',
            'Czech Republic': 'Europe',
            'Czechoslovakia': 'Europe',

            # Asia
            'Japan': 'Asia',
            'South Korea': 'Asia',
            'Korea': 'Asia',
            'China': 'Asia',
            'Hong Kong': 'Asia',
            'India': 'Asia',
            'Thailand': 'Asia',
            'Singapore': 'Asia',
            'Malaysia': 'Asia',
            'Indonesia': 'Asia',
            'Philippines': 'Asia',
            'Taiwan': 'Asia',
            'Vietnam': 'Asia',

            # Latin America
            'Mexico': 'Latin America',
            'Brazil': 'Latin America',
            'Argentina': 'Latin America',
            'Chile': 'Latin America',
            'Colombia': 'Latin America',
            'Peru': 'Latin America',
            'Venezuela': 'Latin America',

            # Oceania
            'Australia': 'Oceania',
            'New Zealand': 'Oceania',

            # Africa
            'South Africa': 'Africa',
            'Egypt': 'Africa',
            'Nigeria': 'Africa',
        }

        # Weighting multipliers for different scenarios
        self.weights = {
            'same_region': 1.5,      # Movie from user's region
            'hollywood': 1.3,         # Hollywood movies (popular globally)
            'different_region': 0.9,  # Movies from other regions
            'asian_cinema': 1.2,      # Asian cinema boost (globally appreciated)
        }

    def get_movie_region(self, movie_country: str) -> str:
        """
        Get region for a movie based on its country

        Args:
            movie_country: Country where movie was produced

        Returns:
            Region string
        """
        # Handle multiple countries (e.g., "USA, UK")
        if pd.isna(movie_country) or movie_country == 'Unknown':
            return 'Other'

        countries = [c.strip() for c in str(movie_country).split(',')]

        # Use first country for classification
        main_country = countries[0]

        return self.country_to_region.get(main_country, 'Other')

    def is_hollywood(self, movie_country: str) -> bool:
        """
        Check if movie is Hollywood/mainstream US production

        Args:
            movie_country: Country where movie was produced

        Returns:
            True if Hollywood movie
        """
        if pd.isna(movie_country):
            return False

        return 'USA' in str(movie_country) or 'United States' in str(movie_country)

    def calculate_weight(self, user_region: str, movie_country: str) -> float:
        """
        Calculate popularity weight for a movie based on user region

        Args:
            user_region: User's region (from phone number)
            movie_country: Movie's production country

        Returns:
            Weight multiplier (typically 0.9 - 1.5)
        """
        # Default weight
        base_weight = 1.0

        movie_region = self.get_movie_region(movie_country)

        # STRATEGY 1: Same region boost
        if user_region == movie_region and user_region != 'Other':
            return self.weights['same_region']

        # STRATEGY 2: Hollywood boost (except for Asian users who prefer variety)
        if self.is_hollywood(movie_country):
            if user_region in ['North America', 'Europe', 'Latin America', 'Oceania']:
                return self.weights['hollywood']
            elif user_region == 'Asia':
                return 1.1  # Slight boost but not as much
            else:
                return 1.0

        # STRATEGY 3: Asian cinema global appreciation
        if movie_region == 'Asia':
            return self.weights['asian_cinema']

        # STRATEGY 4: Different region
        if movie_region != 'Other' and user_region != 'Other':
            return self.weights['different_region']

        return base_weight

    def apply_region_weighting(
        self,
        movies: pd.DataFrame,
        user_region: str,
        score_column: str = 'score'
    ) -> pd.DataFrame:
        """
        Apply region-based weighting to movie scores

        Args:
            movies: DataFrame with movies and scores
            user_region: User's region
            score_column: Column name containing scores to weight

        Returns:
            DataFrame with weighted scores in 'weighted_score' column
        """
        if user_region == 'Other' or user_region == 'Unknown':
            # No weighting for unknown regions
            movies['weighted_score'] = movies[score_column]
            movies['region_weight'] = 1.0
            return movies

        # Calculate weights for each movie
        movies['region_weight'] = movies['country'].apply(
            lambda country: self.calculate_weight(user_region, country)
        )

        # Apply weights to scores
        movies['weighted_score'] = movies[score_column] * movies['region_weight']

        return movies

    def get_region_info(self, user_region: str) -> str:
        """
        Get human-readable info about region preferences

        Args:
            user_region: User's region

        Returns:
            String describing the region weighting strategy
        """
        if user_region == 'North America':
            return "Prioritizing North American and Hollywood films"
        elif user_region == 'Europe':
            return "Prioritizing European and Hollywood films"
        elif user_region == 'Asia':
            return "Prioritizing Asian cinema and international films"
        elif user_region == 'Latin America':
            return "Prioritizing Latin American and Hollywood films"
        elif user_region == 'Oceania':
            return "Prioritizing Oceanic and Hollywood films"
        elif user_region == 'Africa':
            return "Prioritizing African and international films"
        else:
            return "Showing diverse international selection"


def apply_regional_popularity(
    movies_df: pd.DataFrame,
    user_region: str,
    score_column: str = 'score'
) -> pd.DataFrame:
    """
    Convenience function to apply region weighting

    Args:
        movies_df: Movies DataFrame
        user_region: User's region
        score_column: Score column to weight

    Returns:
        DataFrame with weighted scores
    """
    weighter = RegionWeighting()
    return weighter.apply_region_weighting(movies_df, user_region, score_column)


if __name__ == '__main__':
    # Test region weighting
    weighter = RegionWeighting()

    # Test cases
    print("Testing Region Weighting:")
    print("-" * 60)

    test_cases = [
        ("North America", "USA"),
        ("North America", "France"),
        ("Europe", "UK"),
        ("Europe", "USA"),
        ("Asia", "Japan"),
        ("Asia", "USA"),
        ("Latin America", "Brazil"),
        ("Latin America", "USA"),
    ]

    for user_region, movie_country in test_cases:
        weight = weighter.calculate_weight(user_region, movie_country)
        print(f"User: {user_region:20s} | Movie: {movie_country:15s} | Weight: {weight:.2f}")
