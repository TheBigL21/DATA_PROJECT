"""
GENRE ANALYTICS TRACKER

ML-powered adaptive learning system that tracks genre selection frequency
and automatically swaps underperforming genres with better alternatives.

Features:
- Tracks (evening_type, genre) selection counts
- Calculates click-through rates
- Identifies underperforming genres
- Suggests optimal genre swaps
- Auto-updates genre allocation based on user behavior
"""

from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple


class GenreAnalytics:
    """Track and optimize genre selections using ML"""

    def __init__(self, analytics_path: Path = None, config_path: Path = None):
        """
        Initialize genre analytics tracker

        Args:
            analytics_path: Path to analytics JSON file
            config_path: Path to genre allocation config
        """
        if analytics_path is None:
            # Default to data/raw for analytics, compute repo root
            repo_root = Path(__file__).parent.parent.parent
            analytics_path = repo_root / "data" / "raw" / "genre_analytics.json"
        if config_path is None:
            # Default to src/config
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "src" / "config" / "genre_allocation.json"

        self.analytics_path = analytics_path
        self.config_path = config_path

        # Create data directory if needed
        self.analytics_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize analytics data
        self.analytics = self._load_analytics()

        # Load genre allocation config
        self.config = self._load_config()

    def _load_analytics(self) -> Dict:
        """Load analytics data from file"""
        if self.analytics_path.exists():
            with open(self.analytics_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty analytics
            return {
                "evening_types": {},
                "last_updated": datetime.now().isoformat(),
                "total_selections": 0
            }

    def _save_analytics(self):
        """Save analytics data to file"""
        self.analytics["last_updated"] = datetime.now().isoformat()
        with open(self.analytics_path, 'w') as f:
            json.dump(self.analytics, f, indent=2)

    def _load_config(self) -> Dict:
        """Load genre allocation config"""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _save_config(self):
        """Save updated genre allocation config"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def log_selection(self, evening_type: str, selected_genres: List[str]):
        """
        Log genre selection for ML learning

        Args:
            evening_type: Type of evening selected
            selected_genres: List of 1-2 genres user selected
        """
        # Initialize evening type if not exists
        if evening_type not in self.analytics["evening_types"]:
            self.analytics["evening_types"][evening_type] = {}

        # Increment counts for each selected genre
        for genre in selected_genres:
            if genre not in self.analytics["evening_types"][evening_type]:
                self.analytics["evening_types"][evening_type][genre] = {
                    "selections": 0,
                    "presentations": 0,
                    "first_seen": datetime.now().isoformat()
                }

            self.analytics["evening_types"][evening_type][genre]["selections"] += 1
            self.analytics["evening_types"][evening_type][genre]["last_selected"] = datetime.now().isoformat()

        # Increment total
        self.analytics["total_selections"] += len(selected_genres)

        # Save
        self._save_analytics()

    def log_presentation(self, evening_type: str, presented_genres: List[str]):
        """
        Log which genres were presented to user (for CTR calculation)

        Args:
            evening_type: Type of evening
            presented_genres: All genres shown to user (core + extended if clicked)
        """
        # Initialize evening type if not exists
        if evening_type not in self.analytics["evening_types"]:
            self.analytics["evening_types"][evening_type] = {}

        # Increment presentation counts
        for genre in presented_genres:
            if genre not in self.analytics["evening_types"][evening_type]:
                self.analytics["evening_types"][evening_type][genre] = {
                    "selections": 0,
                    "presentations": 0,
                    "first_seen": datetime.now().isoformat()
                }

            self.analytics["evening_types"][evening_type][genre]["presentations"] += 1

        # Save
        self._save_analytics()

    def get_genre_performance(self, evening_type: str) -> Dict[str, float]:
        """
        Calculate click-through rate (CTR) for each genre

        Args:
            evening_type: Type of evening

        Returns:
            Dict mapping genre -> CTR (0.0-1.0)
        """
        if evening_type not in self.analytics["evening_types"]:
            return {}

        performance = {}
        evening_data = self.analytics["evening_types"][evening_type]

        for genre, stats in evening_data.items():
            selections = stats.get("selections", 0)
            presentations = stats.get("presentations", 1)  # Avoid division by zero

            # CTR = selections / presentations
            ctr = selections / presentations if presentations > 0 else 0.0
            performance[genre] = ctr

        return performance

    def identify_underperformers(
        self,
        evening_type: str,
        min_presentations: int = 50,
        threshold_percentile: float = 0.25
    ) -> List[Tuple[str, float]]:
        """
        Identify genres that are underperforming (ML detection)

        Args:
            evening_type: Type of evening
            min_presentations: Minimum presentations before considering swap
            threshold_percentile: Bottom percentile to consider underperforming

        Returns:
            List of (genre, ctr) tuples for underperformers
        """
        performance = self.get_genre_performance(evening_type)

        if not performance:
            return []

        # Filter to genres with sufficient data
        evening_data = self.analytics["evening_types"].get(evening_type, {})
        qualified_genres = {
            genre: ctr
            for genre, ctr in performance.items()
            if evening_data[genre].get("presentations", 0) >= min_presentations
        }

        if len(qualified_genres) < 4:  # Need enough data
            return []

        # Calculate threshold (bottom 25th percentile)
        ctrs = sorted(qualified_genres.values())
        threshold_idx = int(len(ctrs) * threshold_percentile)
        threshold = ctrs[threshold_idx] if threshold_idx < len(ctrs) else 0

        # Identify underperformers
        underperformers = [
            (genre, ctr)
            for genre, ctr in qualified_genres.items()
            if ctr <= threshold
        ]

        return sorted(underperformers, key=lambda x: x[1])  # Sort by CTR ascending

    def suggest_swaps(
        self,
        evening_type: str,
        available_genres: List[str],
        max_swaps: int = 2
    ) -> List[Tuple[str, str, str]]:
        """
        Suggest genre swaps using ML insights

        Args:
            evening_type: Type of evening
            available_genres: All available genres in database
            max_swaps: Maximum number of swaps to suggest

        Returns:
            List of (old_genre, new_genre, reason) tuples
        """
        # Get underperformers
        underperformers = self.identify_underperformers(evening_type)

        if not underperformers:
            return []

        # Get current allocations
        current_core = set(self.config[evening_type]["core"])
        current_extended = set(self.config[evening_type]["extended"])
        current_all = current_core | current_extended

        # Find available alternatives
        alternatives = [g for g in available_genres if g not in current_all]

        # Get performance of alternatives from other evening types
        alternative_scores = {}
        for alt_genre in alternatives:
            # Average CTR across all evening types where it's used
            ctrs = []
            for et in self.analytics["evening_types"]:
                if alt_genre in self.analytics["evening_types"][et]:
                    perf = self.get_genre_performance(et)
                    if alt_genre in perf:
                        ctrs.append(perf[alt_genre])

            if ctrs:
                alternative_scores[alt_genre] = sum(ctrs) / len(ctrs)
            else:
                alternative_scores[alt_genre] = 0.5  # Neutral score for untested

        # Sort alternatives by score
        sorted_alternatives = sorted(
            alternative_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Generate swap suggestions
        suggestions = []
        for (old_genre, old_ctr) in underperformers[:max_swaps]:
            if sorted_alternatives:
                new_genre, new_ctr = sorted_alternatives.pop(0)
                reason = f"Low CTR {old_ctr:.2%} → Better alternative {new_ctr:.2%}"
                suggestions.append((old_genre, new_genre, reason))

        return suggestions

    def apply_swaps(self, evening_type: str, swaps: List[Tuple[str, str]]):
        """
        Apply genre swaps to configuration

        Args:
            evening_type: Type of evening
            swaps: List of (old_genre, new_genre) tuples
        """
        for old_genre, new_genre in swaps:
            # Check which list the old genre is in
            if old_genre in self.config[evening_type]["core"]:
                idx = self.config[evening_type]["core"].index(old_genre)
                self.config[evening_type]["core"][idx] = new_genre
            elif old_genre in self.config[evening_type]["extended"]:
                idx = self.config[evening_type]["extended"].index(old_genre)
                self.config[evening_type]["extended"][idx] = new_genre

        # Save updated config
        self._save_config()

    def get_stats_summary(self) -> Dict:
        """Get summary statistics for monitoring"""
        summary = {
            "total_selections": self.analytics.get("total_selections", 0),
            "last_updated": self.analytics.get("last_updated", "Never"),
            "evening_types": {}
        }

        for evening_type in self.analytics.get("evening_types", {}):
            performance = self.get_genre_performance(evening_type)
            if performance:
                avg_ctr = sum(performance.values()) / len(performance)
                summary["evening_types"][evening_type] = {
                    "genres_tracked": len(performance),
                    "avg_ctr": f"{avg_ctr:.2%}",
                    "top_genre": max(performance.items(), key=lambda x: x[1])[0] if performance else None
                }

        return summary


def main():
    """Demo/test of genre analytics system"""
    print("\n" + "="*60)
    print("GENRE ANALYTICS SYSTEM - DEMO")
    print("="*60)

    tracker = GenreAnalytics()

    # Simulate some selections
    print("\n1. Simulating user selections...")
    tracker.log_presentation("Date night", ["romance", "comedy", "drama", "thriller", "fantasy", "adventure"])
    tracker.log_selection("Date night", ["romance", "comedy"])

    tracker.log_presentation("Friends night", ["comedy", "action", "horror", "thriller", "sci-fi", "adventure"])
    tracker.log_selection("Friends night", ["action", "comedy"])

    # Get statistics
    print("\n2. Current statistics:")
    stats = tracker.get_stats_summary()
    print(json.dumps(stats, indent=2))

    print("\n✓ Genre analytics system initialized successfully")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
