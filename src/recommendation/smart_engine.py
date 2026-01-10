"""
SMART RECOMMENDATION ENGINE (Option C)

New recommendation system with:
1. Option C quality scoring (context-aware, probabilistic)
2. Simplified user flow (evening type → genres → era)
3. Smart keyword matching
4. Era favorability (not filtering)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import sys
import logging
import sqlite3
import json
from datetime import datetime

# Use package imports from src
from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.keyword_analyzer import KeywordAnalyzer, calculate_keyword_match_score
from src.recommendation.inference_engine import (
    build_user_model,
    calculate_popularity_boost,
    infer_pacing_from_session
)
from src.recommendation.time_period_filter import TimePeriodFilter
from src.recommendation.source_material_filter import SourceMaterialFilter
from src.recommendation.age_keywords import calculate_age_score
from src.recommendation.pacing_calculator import calculate_pacing_match


class PersistentFeedbackLearner:
    """Feedback learner that persists to SQLite database - remembers across ALL sessions"""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create user_feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                context_genres TEXT,
                context_era TEXT,
                context_themes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                position_in_session INTEGER,
                previous_movie_id INTEGER
            )
        """)
        
        # Create user_preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                liked_genres TEXT,
                liked_eras TEXT,
                liked_keywords TEXT,
                rejected_genres TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create movie_statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_statistics (
                movie_id INTEGER PRIMARY KEY,
                total_shown INTEGER DEFAULT 0,
                total_liked INTEGER DEFAULT 0,
                total_rejected INTEGER DEFAULT 0,
                avg_satisfaction REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id 
            ON user_feedback(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp 
            ON user_feedback(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_feedback_user_session 
            ON user_feedback(user_id, session_id)
        """)
        
        # Create selection_movie_compatibility table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS selection_movie_compatibility (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                selection_signature TEXT NOT NULL,
                compatibility_score REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, movie_id, selection_signature)
            )
        """)
        
        # Create indexes for compatibility table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compatibility_user_selection 
            ON selection_movie_compatibility(user_id, selection_signature)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compatibility_movie 
            ON selection_movie_compatibility(movie_id)
        """)
        
        # Create users table (for login/signup tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                display_code TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Ensure default user 1000 exists
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, display_code)
            VALUES (1000, '1000')
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_display_code 
            ON users(display_code)
        """)
        
        # Create global_context_movie_fit table (across all users, no user_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_context_movie_fit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                selection_signature TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                shown_count INTEGER DEFAULT 0,
                yes_count INTEGER DEFAULT 0,
                no_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(selection_signature, movie_id)
            )
        """)
        
        # Create indexes for global context fit table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_fit_selection 
            ON global_context_movie_fit(selection_signature)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_fit_movie 
            ON global_context_movie_fit(movie_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_fit_selection_movie 
            ON global_context_movie_fit(selection_signature, movie_id)
        """)
        
        conn.commit()
        conn.close()
    
    def log_feedback(
        self,
        user_id: int,
        session_id: str,
        movie_id: int,
        action: str,
        context: Dict,
        position_in_session: int = 0,
        previous_movie_id: Optional[int] = None
    ):
        """Log user feedback to database (persists across sessions)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Normalize action
        normalized_action = self._normalize_action(action)
        
        cursor.execute("""
            INSERT INTO user_feedback 
            (user_id, session_id, movie_id, action, context_genres, 
             context_era, context_themes, position_in_session, previous_movie_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            session_id,
            movie_id,
            normalized_action,
            json.dumps(context.get('genres', [])),
            context.get('era', 'any'),
            json.dumps(context.get('themes', [])),
            position_in_session,
            previous_movie_id
        ))
        
        # Update movie statistics
        is_positive = normalized_action in ['right', 'up']
        is_negative = normalized_action in ['left']
        
        cursor.execute("""
            INSERT INTO movie_statistics (movie_id, total_shown, total_liked, total_rejected)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(movie_id) DO UPDATE SET
                total_shown = total_shown + 1,
                total_liked = total_liked + ?,
                total_rejected = total_rejected + ?,
                last_updated = CURRENT_TIMESTAMP
        """, (
            movie_id,
            1 if is_positive else 0,
            1 if is_negative else 0,
            1 if is_positive else 0,
            1 if is_negative else 0
        ))
        
        conn.commit()
        conn.close()
        
        # Update user preferences (async or batch - can be optimized)
        self._update_user_preferences(user_id)
        
        # Update compatibility score for this selection-movie combination (per-user)
        self.update_compatibility_score(
            user_id=user_id,
            movie_id=movie_id,
            genres=context.get('genres', []),
            era=context.get('era', 'any'),
            themes=context.get('themes', []),
            action=action
        )
        
        # Update global context-movie fit (across all users)
        self.update_global_context_fit(
            movie_id=movie_id,
            genres=context.get('genres', []),
            era=context.get('era', 'any'),
            themes=context.get('themes', []),
            action=action
        )
    
    def get_user_history(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get user's feedback history from database (across ALL sessions)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            # Get current session + recent history
            cursor.execute("""
                SELECT movie_id, action, timestamp, context_genres, context_era, context_themes
                FROM user_feedback
                WHERE user_id = ? AND (session_id = ? OR timestamp > datetime('now', '-7 days'))
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, session_id, limit))
        else:
            # Get all recent history
            cursor.execute("""
                SELECT movie_id, action, timestamp, context_genres, context_era, context_themes
                FROM user_feedback
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'movie_id': row[0],
                'action': row[1],
                'timestamp': row[2],
                'context': {
                    'genres': json.loads(row[3]) if row[3] else [],
                    'era': row[4],
                    'themes': json.loads(row[5]) if row[5] else []
                }
            })
        
        conn.close()
        return results
    
    def get_adjustment(
        self,
        movie_id: int,
        user_id: int,
        session_id: Optional[str],
        user_history: List[Dict],
        content_similarity_model,
        graph_model,
        current_genres: List[str],
        current_era: str,
        min_context_observations: int = 5
    ) -> float:
        """
        Get feedback adjustment [-0.10, +0.10] from persistent user history.
        
        IMPORTANT: Only uses history from the SAME primary context (genres+era).
        This prevents preference leakage across different genre/era combinations.
        
        Args:
            movie_id: Movie ID to score
            user_id: User ID
            session_id: Session ID (optional)
            user_history: Full user history from database
            content_similarity_model: Content similarity model
            graph_model: Graph co-occurrence model
            current_genres: Current selected genres (for context filtering)
            current_era: Current selected era (for context filtering)
            min_context_observations: Minimum interactions in this context before applying adjustment (default: 5)
        
        Returns:
            Adjustment score in [-0.10, +0.10]. Returns 0.0 if insufficient context data.
        """
        if not user_history:
            return 0.0
        
        # Filter history to ONLY items matching current primary context (genres+era)
        primary_context_sig = self._create_primary_context_signature(current_genres, current_era)
        
        context_matched_history = []
        for item in user_history:
            # Extract context from history item
            context = item.get('context', {})
            item_genres = context.get('genres', []) if isinstance(context, dict) else []
            item_era = context.get('era', 'any') if isinstance(context, dict) else 'any'
            
            # Create signature for this history item
            item_context_sig = self._create_primary_context_signature(item_genres, item_era)
            
            # Only include if contexts match
            if item_context_sig == primary_context_sig:
                context_matched_history.append(item)
        
        # Minimum observations threshold: need at least N interactions in this context
        if len(context_matched_history) < min_context_observations:
            return 0.0  # Not enough data in this context - return neutral
        
        # Use all matching history (not just 20) since we've already filtered by context
        # Limit to 200 most recent to avoid excessive computation
        recent_feedback = context_matched_history[:200]
        
        if not recent_feedback:
            return 0.0
        
        # Bucket weighting parameters
        BUCKET_SIZE = 20  # Number of interactions per bucket
        GAMMA = 0.92  # Decay factor per bucket (slight decay, stable preferences)
        
        def get_bucket_weight(index: int) -> float:
            """
            Compute bucket weight for item at given index.
            
            Most recent 20 items (bucket 0): weight = 1.00
            Next 20 items (bucket 1): weight = 0.92
            Next 20 items (bucket 2): weight = 0.92^2 = 0.846
            etc.
            
            Args:
                index: Index in recent_feedback (0 = most recent)
            
            Returns:
                Weight in [0, 1] (decaying with bucket number)
            """
            bucket_number = index // BUCKET_SIZE
            return GAMMA ** bucket_number
        
        boost = 0.0
        penalty = 0.0
        
        # Separate liked and rejected for efficient neighbor caching
        liked_entries = []  # (index, movie_id, action)
        rejected_entries = []  # (index, movie_id)
        
        for idx, item in enumerate(recent_feedback):
            action = item.get('action', '')
            item_movie_id = item.get('movie_id')
            
            if action in ['right', 'up', 'yes', 'final']:
                liked_entries.append((idx, item_movie_id, action))
            elif action in ['left', 'no']:
                rejected_entries.append((idx, item_movie_id))
        
        if not liked_entries and not rejected_entries:
            return 0.0
        
        # Pre-compute graph neighbors for all liked/rejected movies (use all, not just last 20)
        liked_neighbors_cache = {}
        rejected_neighbors_cache = {}
        
        if graph_model:
            for idx, liked_id, action in liked_entries:
                if liked_id not in liked_neighbors_cache:
                    try:
                        liked_neighbors_cache[liked_id] = dict(
                            graph_model.get_neighbors(liked_id, top_k=100)
                        )
                    except:
                        liked_neighbors_cache[liked_id] = {}
            
            for idx, rejected_id in rejected_entries:
                if rejected_id not in rejected_neighbors_cache:
                    try:
                        rejected_neighbors_cache[rejected_id] = dict(
                            graph_model.get_neighbors(rejected_id, top_k=100)
                        )
                    except:
                        rejected_neighbors_cache[rejected_id] = {}
        
        # Calculate similarity to liked movies with bucket weighting
        for idx, liked_id, action_type in liked_entries:
            # Get bucket weight based on position in recent_feedback
            weight = get_bucket_weight(idx)
            
            # Compute similarity (60% content, 40% graph)
            sim_content = 0.0
            if content_similarity_model:
                try:
                    sim_content = content_similarity_model.get_similarity(movie_id, liked_id)
                except:
                    sim_content = 0.0
            
            sim_graph = liked_neighbors_cache.get(liked_id, {}).get(movie_id, 0.0) if graph_model else 0.0
            sim = 0.6 * sim_content + 0.4 * sim_graph
            
            # Apply bucket weight to similarity before computing boost
            weighted_sim = weight * sim
            
            if action_type in ['right', 'yes']:
                boost = max(boost, 0.08 * weighted_sim)
            elif action_type in ['up', 'final']:
                boost = max(boost, 0.10 * weighted_sim)
        
        # Calculate similarity to rejected movies with bucket weighting
        for idx, rejected_id in rejected_entries:
            # Get bucket weight based on position in recent_feedback
            weight = get_bucket_weight(idx)
            
            sim_content = 0.0
            if content_similarity_model:
                try:
                    sim_content = content_similarity_model.get_similarity(movie_id, rejected_id)
                except:
                    sim_content = 0.0
            
            sim_graph = rejected_neighbors_cache.get(rejected_id, {}).get(movie_id, 0.0) if graph_model else 0.0
            sim = 0.6 * sim_content + 0.4 * sim_graph
            
            # Apply bucket weight to similarity before computing penalty
            weighted_sim = weight * sim
            penalty = min(penalty, -0.10 * weighted_sim)
        
        return np.clip(boost + penalty, -0.10, +0.10)
    
    def _create_primary_context_signature(
        self,
        genres: List[str],
        era: str
    ) -> str:
        """
        Create primary context signature (genres + era only).
        
        This is used to scope learning adjustments to the same context,
        preventing preference leakage across different genre/era combinations.
        
        Format: "genre1|genre2|era"
        - Genres sorted alphabetically
        - Era normalized to lowercase
        
        Examples:
        - genres=["action", "dystopia"], era="new_era"
          → "action|dystopia|new_era"
        
        - genres=["comedy"], era="millennium"
          → "comedy|millennium"
        
        Args:
            genres: List of genre strings
            era: Era string ('new_era', 'millennium', etc.)
            
        Returns:
            Primary context signature string
        """
        # Sort and normalize genres
        sorted_genres = sorted([g.lower().strip() for g in genres if g])
        genre_str = '|'.join(sorted_genres) if sorted_genres else ''
        
        # Normalize era
        era_str = era.lower().strip() if era else 'any'
        
        # Combine: genres|era
        return f"{genre_str}|{era_str}"
    
    def _create_selection_signature(
        self, 
        genres: List[str], 
        era: str, 
        themes: List[str]
    ) -> str:
        """
        Create deterministic signature for selection combination (full context).
        
        Format: "genre1|genre2|era|theme1|theme2"
        - Genres sorted alphabetically
        - Themes sorted alphabetically
        - Era normalized to lowercase
        - Pipe-separated for easy parsing
        
        Examples:
        - genres=["action", "thriller"], era="new_era", themes=["espionage"]
          → "action|thriller|new_era|espionage"
        
        - genres=["comedy"], era="any", themes=[]
          → "comedy|any|"
        
        Args:
            genres: List of genre strings
            era: Era string ('new_era', 'millennium', etc.)
            themes: List of theme strings
            
        Returns:
            Signature string (deterministic, sortable)
        """
        # Sort and normalize genres
        sorted_genres = sorted([g.lower().strip() for g in genres if g])
        genre_str = '|'.join(sorted_genres) if sorted_genres else ''
        
        # Normalize era
        era_str = era.lower().strip() if era else 'any'
        
        # Sort and normalize themes
        sorted_themes = sorted([t.lower().strip() for t in themes if t])
        theme_str = '|'.join(sorted_themes) if sorted_themes else ''
        
        # Combine: genres|era|themes
        return f"{genre_str}|{era_str}|{theme_str}"
    
    def _normalize_action(self, action: str) -> str:
        """Normalize action types: yes=right, final=up, no=left"""
        action_map = {
            'yes': 'right',
            'final': 'up',
            'no': 'left'
        }
        return action_map.get(action.lower(), action.lower())
    
    def _update_user_preferences(self, user_id: int):
        """Update user preference profile from feedback history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all feedback for this user (last 100 actions)
        cursor.execute("""
            SELECT movie_id, action, context_genres, context_era, context_themes
            FROM user_feedback
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 100
        """, (user_id,))
        
        genre_weights = {}
        era_weights = {}
        keyword_weights = {}
        
        for row in cursor.fetchall():
            movie_id, action, genres_json, era, themes_json = row
            genres = json.loads(genres_json) if genres_json else []
            themes = json.loads(themes_json) if themes_json else []
            
            normalized_action = self._normalize_action(action)
            
            # Weight by action type
            if normalized_action in ['right', 'up']:
                weight = 0.1 if normalized_action == 'right' else 0.15
                for genre in genres:
                    genre_weights[genre] = genre_weights.get(genre, 0.0) + weight
                if era:
                    era_weights[era] = era_weights.get(era, 0.0) + weight
                for theme in themes:
                    keyword_weights[theme] = keyword_weights.get(theme, 0.0) + weight
            elif normalized_action in ['left']:
                weight = -0.1
                for genre in genres:
                    genre_weights[genre] = genre_weights.get(genre, 0.0) + weight
        
        # Normalize weights
        max_genre = max(abs(v) for v in genre_weights.values()) if genre_weights else 1.0
        max_era = max(abs(v) for v in era_weights.values()) if era_weights else 1.0
        max_keyword = max(abs(v) for v in keyword_weights.values()) if keyword_weights else 1.0
        
        genre_weights = {k: v / max_genre for k, v in genre_weights.items()} if max_genre > 0 else {}
        era_weights = {k: v / max_era for k, v in era_weights.items()} if max_era > 0 else {}
        keyword_weights = {k: v / max_keyword for k, v in keyword_weights.items()} if max_keyword > 0 else {}
        
        # Save to database
        cursor.execute("""
            INSERT INTO user_preferences 
            (user_id, liked_genres, liked_eras, liked_keywords, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                liked_genres = ?,
                liked_eras = ?,
                liked_keywords = ?,
                last_updated = CURRENT_TIMESTAMP
        """, (
            user_id,
            json.dumps(genre_weights),
            json.dumps(era_weights),
            json.dumps(keyword_weights),
            json.dumps(genre_weights),
            json.dumps(era_weights),
            json.dumps(keyword_weights)
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get user's learned preferences from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT liked_genres, liked_eras, liked_keywords
            FROM user_preferences
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'genres': json.loads(row[0]) if row[0] else {},
                'eras': json.loads(row[1]) if row[1] else {},
                'keywords': json.loads(row[2]) if row[2] else {}
            }
        return {'genres': {}, 'eras': {}, 'keywords': {}}
    
    def update_compatibility_score(
        self,
        user_id: int,
        movie_id: int,
        genres: List[str],
        era: str,
        themes: List[str],
        action: str  # 'right', 'up', 'left', 'no', 'yes', 'final'
    ):
        """
        Update compatibility score for movie-selection combination.
        
        Update Rules:
        - YES (right, yes): +0.1 per action
        - YES (up, final): +0.1 per action (same weight)
        - NO (left, no): -0.1 per action
        - Scores accumulate: Multiple YES/NO add up
        - Clamped to [-1.0, +1.0]
        - No time decay: All feedback equally weighted
        
        Examples:
        - User says YES once: score = 0.0 + 0.1 = 0.1
        - User says YES twice: score = 0.1 + 0.1 = 0.2
        - User says NO once: score = 0.2 - 0.1 = 0.1
        - User says NO three times: score = 0.1 - 0.3 = -0.2
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            genres: List of genres from selection
            era: Era from selection
            themes: List of themes from selection
            action: User action ('yes', 'no', 'right', 'left', 'up', 'final')
        """
        selection_sig = self._create_selection_signature(genres, era, themes)
        normalized_action = self._normalize_action(action)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current score
        cursor.execute("""
            SELECT compatibility_score, interaction_count
            FROM selection_movie_compatibility
            WHERE user_id = ? AND movie_id = ? AND selection_signature = ?
        """, (user_id, movie_id, selection_sig))
        
        row = cursor.fetchone()
        
        if row:
            current_score = row[0]
            interaction_count = row[1]
        else:
            current_score = 0.0
            interaction_count = 0
        
        # Calculate score adjustment
        if normalized_action in ['right', 'up', 'yes', 'final']:
            # Positive feedback: +0.1 (all YES actions have same weight)
            adjustment = 0.1
            new_score = min(1.0, current_score + adjustment)
        elif normalized_action in ['left', 'no']:
            # Negative feedback: -0.1
            adjustment = -0.1
            new_score = max(-1.0, current_score + adjustment)
        else:
            # Unknown action, no change
            new_score = current_score
            adjustment = 0.0
        
        # Update or insert
        cursor.execute("""
            INSERT INTO selection_movie_compatibility 
            (user_id, movie_id, selection_signature, compatibility_score, interaction_count, last_interaction)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, movie_id, selection_signature) DO UPDATE SET
                compatibility_score = ?,
                interaction_count = interaction_count + 1,
                last_interaction = CURRENT_TIMESTAMP
        """, (
            user_id, movie_id, selection_sig, new_score, interaction_count + 1,
            new_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_compatibility_score(
        self,
        user_id: int,
        movie_id: int,
        genres: List[str],
        era: str,
        themes: List[str]
    ) -> float:
        """
        Get compatibility score for movie-selection combination.
        
        Returns:
            Compatibility score in [-1.0, +1.0]
            - Positive: Movie fits well with this selection (user liked it)
            - Negative: Movie doesn't fit well (user rejected it)
            - 0.0: No data (neutral, no learning yet)
        
        Examples:
        - Score = +0.3: User liked this movie 3 times for this selection
        - Score = -0.2: User rejected this movie 2 times for this selection
        - Score = 0.0: No feedback yet for this movie-selection combination
        """
        selection_sig = self._create_selection_signature(genres, era, themes)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT compatibility_score
            FROM selection_movie_compatibility
            WHERE user_id = ? AND movie_id = ? AND selection_signature = ?
        """, (user_id, movie_id, selection_sig))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return float(row[0])
        return 0.0  # No data, neutral score
    
    def update_global_context_fit(
        self,
        movie_id: int,
        genres: List[str],
        era: str,
        themes: List[str],
        action: str
    ):
        """
        Update global context-movie fit statistics (across all users).
        
        This learns which movies work well for which questionnaire contexts
        based on aggregated YES/NO feedback from all users.
        
        Args:
            movie_id: Movie ID
            genres: List of genres from selection
            era: Era from selection
            themes: List of themes from selection
            action: User action ('yes', 'no', 'right', 'left', 'up', 'final')
        """
        selection_sig = self._create_selection_signature(genres, era, themes)
        normalized_action = self._normalize_action(action)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current stats
        cursor.execute("""
            SELECT shown_count, yes_count, no_count
            FROM global_context_movie_fit
            WHERE selection_signature = ? AND movie_id = ?
        """, (selection_sig, movie_id))
        
        row = cursor.fetchone()
        
        if row:
            shown_count, yes_count, no_count = row
        else:
            shown_count, yes_count, no_count = 0, 0, 0
        
        # Update stats based on action
        new_shown_count = shown_count + 1
        
        if normalized_action in ['right', 'up', 'yes', 'final']:
            # Positive feedback: increment yes_count
            new_yes_count = yes_count + 1
            new_no_count = no_count
        elif normalized_action in ['left', 'no']:
            # Negative feedback: increment no_count
            new_yes_count = yes_count
            new_no_count = no_count + 1
        else:
            # Unknown action, only increment shown
            new_yes_count = yes_count
            new_no_count = no_count
        
        # Update or insert
        cursor.execute("""
            INSERT INTO global_context_movie_fit 
            (selection_signature, movie_id, shown_count, yes_count, no_count, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(selection_signature, movie_id) DO UPDATE SET
                shown_count = ?,
                yes_count = ?,
                no_count = ?,
                last_updated = CURRENT_TIMESTAMP
        """, (
            selection_sig, movie_id, new_shown_count, new_yes_count, new_no_count,
            new_shown_count, new_yes_count, new_no_count
        ))
        
        conn.commit()
        conn.close()
    
    def get_global_context_fit_score(
        self,
        movie_id: int,
        genres: List[str],
        era: str,
        themes: List[str],
        min_observations: int = 5,
        alpha: float = 2.0,
        beta: float = 2.0
    ) -> float:
        """
        Get global context-movie fit score (across all users).
        
        Uses Beta smoothing to avoid noise from small sample sizes:
        p = (yes_count + alpha) / (shown_count + alpha + beta)
        
        Then converts to adjustment centered at 0:
        score = p - 0.5  (range [-0.5, +0.5])
        
        Args:
            movie_id: Movie ID
            genres: List of genres from selection
            era: Era from selection
            themes: List of themes from selection
            min_observations: Minimum shown_count to apply score (default: 5)
            alpha: Beta prior parameter for smoothing (default: 2.0)
            beta: Beta prior parameter for smoothing (default: 2.0)
        
        Returns:
            Global fit score in [-0.5, +0.5]
            - Positive: Movie tends to be liked in this context (across users)
            - Negative: Movie tends to be rejected in this context (across users)
            - 0.0: No data or insufficient observations (neutral)
        """
        selection_sig = self._create_selection_signature(genres, era, themes)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT shown_count, yes_count, no_count
            FROM global_context_movie_fit
            WHERE selection_signature = ? AND movie_id = ?
        """, (selection_sig, movie_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return 0.0  # No data
        
        shown_count, yes_count, no_count = row
        
        # Need minimum observations to trust the score
        if shown_count < min_observations:
            return 0.0
        
        # Beta smoothing: p = (yes + alpha) / (shown + alpha + beta)
        # This prevents small samples from dominating (e.g., 1 yes out of 1 shown → not 100%)
        smoothed_probability = (yes_count + alpha) / (shown_count + alpha + beta)
        
        # Convert to adjustment centered at 0.5: score = p - 0.5
        # Range: [0 - 0.5, 1 - 0.5] = [-0.5, +0.5]
        global_score = smoothed_probability - 0.5
        
        return float(global_score)
    
    def user_exists(self, user_id: int) -> bool:
        """Check if a user exists in the users table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM users WHERE user_id = ?
        """, (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_user_by_display_code(self, display_code: str) -> Optional[int]:
        """Get user_id by display_code (4-digit string). Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id FROM users WHERE display_code = ?
        """, (display_code,))
        
        row = cursor.fetchone()
        conn.close()
        
        return int(row[0]) if row else None
    
    def create_user(self, user_id: int, display_code: str) -> bool:
        """
        Create a new user. Returns True if created, False if already exists.
        
        Args:
            user_id: Internal user ID (4-digit, 1000-9999)
            display_code: Display code (4-digit string, e.g. "1234")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO users (user_id, display_code, last_active)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (user_id, display_code))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # User already exists (user_id or display_code conflict)
            conn.rollback()
            conn.close()
            return False
    
    def find_next_available_user_id(self, start_from: int = 1001, max_id: int = 9999) -> Optional[int]:
        """
        Find next available user_id in range [start_from, max_id].
        Returns None if no available ID found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id FROM users 
            WHERE user_id >= ? AND user_id <= ?
            ORDER BY user_id
        """, (start_from, max_id))
        
        existing_ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        # Find first gap or next after highest
        for candidate_id in range(start_from, max_id + 1):
            if candidate_id not in existing_ids:
                return candidate_id
        
        return None


class SmartRecommendationEngine:
    """
    Smart recommendation engine with Option C quality scoring and keyword matching.
    """

    # Vote confidence constants
    MAX_VOTES_NORMALIZATION = 2_000_000  # Reference point for vote confidence
    MIN_CONFIDENCE = 0.3  # Minimum confidence multiplier
    MAX_CONFIDENCE = 1.0  # Maximum confidence multiplier

    # Era configurations now use TimePeriodFilter (no local config needed)

    # Evening type modifiers for quality scoring
    EVENING_MODIFIERS = {
        'date_night': 1.2,      # Want impressive movies
        'family_night': 1.1,    # Want reliable entertainment
        'friends_night': 1.0,   # Social focus, forgiving
        'chill_evening': 0.9    # Experimental mood
    }

    def __init__(
        self,
        base_engine: RecommendationEngine,
        keyword_analyzer: Optional[KeywordAnalyzer] = None,
        content_similarity: Optional[Any] = None,
        feedback_db_path: Optional[Path] = None
    ):
        """
        Initialize smart recommendation engine.

        Args:
            base_engine: Base engine with CF and graph models
            keyword_analyzer: Optional keyword analyzer for suggestions
            content_similarity: Optional ContentSimilarity model
            feedback_db_path: Optional path to feedback database
        """
        self.base_engine = base_engine
        self.movies = base_engine.movies
        self.keyword_analyzer = keyword_analyzer
        self.content_similarity = content_similarity
        
        # Initialize persistent feedback learner
        if feedback_db_path:
            self.feedback_learner = PersistentFeedbackLearner(feedback_db_path)
        else:
            self.feedback_learner = None

        # Base scoring weights (when NO keywords selected)
        self.w_genre_base = 0.30
        self.w_cf_base = 0.30
        self.w_graph_base = 0.05
        self.w_quality_base = 0.20
        self.w_session_base = 0.15

        # Scoring weights with keywords
        self.w_genre_kw = 0.25      # Genre matching (primary user intent)
        self.w_keywords = 0.30      # INCREASED - keyword matching is now primary
        self.w_cf_kw = 0.20         # Collaborative filtering
        self.w_quality_kw = 0.15    # Quality score (dynamically adjusted)
        self.w_session_kw = 0.10    # Session similarity

    def _calculate_vote_confidence(self, num_votes: int) -> float:
        """
        Calculate confidence multiplier from vote count.

        Uses logarithmic scaling to reward movies with more votes.
        Clamps result between MIN_CONFIDENCE and MAX_CONFIDENCE.

        Args:
            num_votes: Number of votes for the movie

        Returns:
            Confidence multiplier in range [0.3, 1.0]
        """
        confidence = np.log1p(num_votes) / np.log1p(self.MAX_VOTES_NORMALIZATION)
        return max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, confidence))

    def recommend(
        self,
        user_id: int,
        genres: List[str],
        era: Optional[str] = None,
        source_material: Optional[str] = None,
        themes: Optional[List[str]] = None,
        session_history: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
        top_k: int = 20
    ) -> List[int]:
        """
        Generate recommendations with new filtering system and cross-session learning.

        Args:
            user_id: User ID
            genres: List of 1-2 selected genres
            era: Selected time period ('new_era', 'millennium', 'old_school', 'golden_era', 'any')
            source_material: Source material preference ('book', 'true_story', 'any', etc.)
            themes: List of thematic keywords selected by user
            session_history: List of {movie_id, action, num_votes, pacing_score} dicts (current session)
            session_id: Optional session ID for cross-session learning
            top_k: Number of recommendations to return

        Returns:
            List of movie_ids
        """
        if session_history is None:
            session_history = []
        if themes is None:
            themes = []
        if era is None:
            era = 'any'
        if source_material is None:
            source_material = 'any'

        # IMPORTANT: Keep DB history SEPARATE from session_history
        # session_history = current session only (from frontend)
        # db_history = persistent history (for context-scoped learning only)
        db_history = []
        if self.feedback_learner and user_id is not None:
            try:
                db_history = self.feedback_learner.get_user_history(
                    user_id=user_id,
                    session_id=None,  # Get all sessions, not just current
                    limit=500  # Increased to 500 for better long-term learning (after context filtering)
                )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load user history from database: {e}")

        # Extract session context (CURRENT SESSION ONLY)
        session_positive = [
            item['movie_id'] for item in session_history
            if item.get('action') in ['right', 'up', 'yes', 'final']
        ]
        already_shown = [item['movie_id'] for item in session_history]
        
        # Note: db_history is used separately in _score_candidates_new for context-scoped adjustments

        # STAGE 1: Generate candidates
        candidates = self._generate_candidates(
            user_id=user_id,
            genres=genres,
            era=era,
            quality_floor=6.0,  # Quality floor set to 6.0
            session_positive_movies=session_positive,
            already_shown=already_shown
        )

        # STAGE 2: Score all candidates (strict scoring)
        # Pass db_history separately for context-scoped learning
        scores = self._score_candidates_new(
            candidates=candidates,
            user_id=user_id,
            genres=genres,
            era=era,
            source_material=source_material,
            themes=themes,
            session_positive_movies=session_positive,
            session_history=session_history,  # Current session only (for session-specific signals)
            db_history=db_history  # Separate DB history (for context-scoped persistent learning)
        )

        # Store strict scores for interactive selector (80/20 split)
        self.strict_scores = dict(zip(candidates['movie_id'].tolist(), scores))

        # STAGE 3: Sort and return top K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return candidates.iloc[top_indices]['movie_id'].tolist()

    def update_feedback(
        self,
        user_id: int,
        session_id: str,
        movie_id: int,
        action: str,
        context: Dict,
        position_in_session: int = 0,
        previous_movie_id: Optional[int] = None
    ):
        """
        Update feedback learner and persist to database (remembers across sessions).
        
        Args:
            user_id: User ID
            session_id: Session ID
            movie_id: Movie ID
            action: Action type ('yes', 'no', 'final', 'left', 'right', 'up')
            context: Context dict with 'genres', 'era', 'themes'
            position_in_session: Position in current session
            previous_movie_id: Previous movie ID shown
        """
        if self.feedback_learner:
            try:
                self.feedback_learner.log_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    movie_id=movie_id,
                    action=action,
                    context=context,
                    position_in_session=position_in_session,
                    previous_movie_id=previous_movie_id
                )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to persist feedback: {e}")

    def get_strict_scores(self) -> Dict[int, float]:
        """
        Get strict scores for all candidates (used by InteractiveSelector).

        Returns:
            Dict mapping movie_id → strict score [0.0, 1.0]
        """
        return getattr(self, 'strict_scores', {})

    def _generate_candidates(
        self,
        user_id: int,
        genres: List[str],
        era: str,
        quality_floor: float,
        session_positive_movies: List[int],
        already_shown: List[int]
    ) -> pd.DataFrame:
        """
        Generate candidate movies using HARD filtering (strict requirements).

        HARD FILTERS (non-negotiable):
        - Genre: 
          * For 2 genres: Try strict AND (both genres required), fallback to OR (either genre) if no candidates
          * For 1 genre: EXACT match required
        - Quality: rating >= quality_floor (6.0)
        - Popularity: num_votes >= 5000
        - Era: Movies must be within exact year range (if era != 'any')
          (uses TimePeriodFilter for consistent era definitions)

        Returns 50-200 high-quality candidates.
        """
        genres_lower = [g.lower().strip() for g in genres]
        
        # HARD FILTER 2: Quality floor (6.0 minimum)
        rating_mask = self.movies['avg_rating'] >= quality_floor

        # HARD FILTER 3: Popularity floor (5000 votes minimum)
        popularity_mask = self.movies['num_votes'] >= 5000

        # HARD FILTER 4: Era range (exact year match required)
        if era != 'any':
            year_min, year_max = TimePeriodFilter.get_year_range(era)
            if year_min is not None and year_max is not None:
                era_mask = (self.movies['year'] >= year_min) & (self.movies['year'] <= year_max)
            else:
                era_mask = pd.Series([True] * len(self.movies), index=self.movies.index)
        else:
            era_mask = pd.Series([True] * len(self.movies), index=self.movies.index)

        # HARD FILTER 1: Genre match (with strict/fallback logic for 2 genres)
        if len(genres_lower) == 2:
            # For 2 genres: Try strict AND first (both genres required)
            g1, g2 = genres_lower[0], genres_lower[1]
            
            def matches_both(movie_genres):
                if not isinstance(movie_genres, (list, np.ndarray)):
                    return False
                movie_genres_lower = [str(g).lower().strip() for g in movie_genres]
                return (g1 in movie_genres_lower) and (g2 in movie_genres_lower)
            
            def matches_either(movie_genres):
                if not isinstance(movie_genres, (list, np.ndarray)):
                    return False
                movie_genres_lower = [str(g).lower().strip() for g in movie_genres]
                return (g1 in movie_genres_lower) or (g2 in movie_genres_lower)
            
            # Try strict AND (both genres)
            genre_mask = self.movies['genres'].apply(matches_both)
            candidates = self.movies[genre_mask & rating_mask & popularity_mask & era_mask].copy()
            
            # Remove already shown
            candidates = candidates[~candidates['movie_id'].isin(already_shown)]
            
            # If no candidates left after removing already_shown, fallback to OR (either genre)
            if len(candidates) == 0:
                genre_mask = self.movies['genres'].apply(matches_either)
                candidates = self.movies[genre_mask & rating_mask & popularity_mask & era_mask].copy()
                # Remove already shown again with relaxed filter
                candidates = candidates[~candidates['movie_id'].isin(already_shown)]
        else:
            # For single genre: current behavior (exact match)
            def matches_genre(movie_genres):
                if not isinstance(movie_genres, (list, np.ndarray)):
                    return False
                movie_genres_lower = [str(g).lower().strip() for g in movie_genres]
                return any(genre_lower in movie_genres_lower for genre_lower in genres_lower)
            
            genre_mask = self.movies['genres'].apply(matches_genre)
            candidates = self.movies[genre_mask & rating_mask & popularity_mask & era_mask].copy()
            
            # Remove already shown
            candidates = candidates[~candidates['movie_id'].isin(already_shown)]

        # Limit to top 200 by votes (popularity proxy for quality)
        if len(candidates) > 200:
            candidates = candidates.nlargest(200, 'num_votes')

        return candidates

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        genres: List[str],
        selected_keywords: Dict[str, Any],
        user_model: Dict[str, Any],
        session_positive_movies: List[int],
        session_history: List[Dict]
    ) -> np.ndarray:
        """
        Score all candidates using weighted components with inference.

        Returns:
            Array of scores (one per candidate)
        """
        n_candidates = len(candidates)

        # Determine if keywords were selected
        has_keywords = (
            selected_keywords.get('age') or
            selected_keywords.get('general') or
            selected_keywords.get('adaptation') or
            selected_keywords.get('pacing')
        )

        # Select weights based on keyword usage
        if has_keywords:
            w_genre = self.w_genre_kw
            w_keywords = self.w_keywords
            w_cf = self.w_cf_kw
            w_quality = user_model['quality']['quality_weight']  # Dynamic
            w_session = self.w_session_kw + (0.05 if len(session_history) >= 5 else 0)
            w_graph = 0.05
        else:
            w_genre = self.w_genre_base
            w_keywords = 0.0
            w_cf = self.w_cf_base
            w_quality = self.w_quality_base
            w_session = self.w_session_base
            w_graph = self.w_graph_base

        # Normalize weights
        total_w = w_genre + w_keywords + w_cf + w_quality + w_session + w_graph
        w_genre /= total_w
        w_keywords /= total_w
        w_cf /= total_w
        w_quality /= total_w
        w_session /= total_w
        w_graph /= total_w

        # Calculate component scores
        scores = np.zeros(n_candidates)

        # Component 1: Genre matching
        genre_scores = self._get_genre_scores(candidates, genres)
        scores += w_genre * genre_scores

        # Component 2: Keyword matching (if keywords selected)
        if has_keywords:
            keyword_scores = self._get_keyword_scores_enhanced(
                candidates,
                selected_keywords,
                user_model['pacing']
            )
            scores += w_keywords * keyword_scores

        # Component 3: Collaborative Filtering
        cf_scores = self._get_cf_scores(candidates, user_id)
        scores += w_cf * cf_scores

        # Component 4: Quality score
        quality_scores = self._get_quality_scores_simple(candidates)
        scores += w_quality * quality_scores

        # Component 5: Session similarity (with pacing awareness)
        session_scores = self._get_session_scores_enhanced(
            candidates,
            genres,
            session_positive_movies,
            session_history,
            user_model['pacing']
        )
        scores += w_session * session_scores

        # Component 6: Graph similarity
        graph_scores = self._get_graph_scores(candidates, session_positive_movies)
        scores += w_graph * graph_scores

        return scores

    def _get_genre_scores(self, candidates: pd.DataFrame, selected_genres: List[str]) -> np.ndarray:
        """
        Calculate explicit genre matching scores (0-1 normalized).

        Scoring:
        - Both selected genres match: 1.0
        - One selected genre matches: 0.5
        - No match: 0.0 (but already filtered out in candidate generation)

        Args:
            candidates: DataFrame of candidate movies
            selected_genres: List of 1-2 genres selected by user

        Returns:
            Array of genre match scores (0-1)
        """
        scores = np.zeros(len(candidates))
        selected_set = {g.lower() for g in selected_genres}

        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_genres = {g.lower() for g in row['genres']} if isinstance(row['genres'], (list, np.ndarray)) else set()

            matches = len(movie_genres & selected_set)

            if len(selected_genres) == 1:
                # Single genre: binary match
                scores[idx] = 1.0 if matches > 0 else 0.0
            else:
                # Two genres: 1.0 for both, 0.5 for one, 0.0 for none
                if matches >= 2:
                    scores[idx] = 1.0
                elif matches == 1:
                    scores[idx] = 0.5
                else:
                    scores[idx] = 0.0

        return scores

    def _get_cf_scores(self, candidates: pd.DataFrame, user_id: int) -> np.ndarray:
        """Get collaborative filtering scores (0-1 normalized)."""
        if not hasattr(self.base_engine, 'cf_model') or not self.base_engine.cf_model:
            # No CF model, return neutral
            return np.ones(len(candidates)) * 0.5

        scores = np.zeros(len(candidates))
        for i, movie_id in enumerate(candidates['movie_id']):
            try:
                score = self.base_engine.cf_model.predict(user_id, movie_id)
                scores[i] = (score + 1) / 2  # Normalize from [-1, 1] to [0, 1]
            except Exception:
                scores[i] = 0.5

        return scores

    def _get_graph_scores(self, candidates: pd.DataFrame, session_positive_movies: List[int]) -> np.ndarray:
        """Get graph co-occurrence scores (0-1 normalized)."""
        if not session_positive_movies or not hasattr(self.base_engine, 'graph'):
            return np.zeros(len(candidates))

        scores = np.zeros(len(candidates))
        for i, movie_id in enumerate(candidates['movie_id']):
            max_similarity = 0
            for pos_movie in session_positive_movies[-5:]:  # Last 5 positive
                try:
                    neighbors = dict(self.base_engine.graph.get_neighbors(pos_movie, top_k=100))
                    if movie_id in neighbors:
                        max_similarity = max(max_similarity, neighbors[movie_id])
                except Exception:
                    pass
            scores[i] = min(1.0, max_similarity)

        return scores

    def _get_session_scores(
        self,
        candidates: pd.DataFrame,
        selected_genres: List[str],
        session_positive_movies: List[int]
    ) -> np.ndarray:
        """Get session-based similarity scores."""
        if not session_positive_movies:
            return np.zeros(len(candidates))

        # Get session movies
        session_movies = self.movies[self.movies['movie_id'].isin(session_positive_movies[-5:])]

        scores = np.zeros(len(candidates))
        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Genre overlap
            movie_genres = set(row['genres']) if isinstance(row['genres'], (list, np.ndarray)) else set()
            selected_set = set(selected_genres)
            genre_overlap = len(movie_genres & selected_set) / len(selected_set) if selected_set else 0

            # Year similarity (within 5 years of session movies)
            if len(session_movies) > 0:
                avg_session_year = session_movies['year'].mean()
                year_diff = abs(row['year'] - avg_session_year)
                year_sim = max(0, 1 - year_diff / 20)  # Decay over 20 years
            else:
                year_sim = 0

            scores[idx] = 0.6 * genre_overlap + 0.4 * year_sim

        return scores

    def _get_quality_scores(self, candidates: pd.DataFrame, evening_type: str) -> np.ndarray:
        """
        Calculate Option C quality scores.

        Uses:
        - Base quality (exponential boost)
        - Confidence (vote count)
        - Evening type modifier
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Get combined rating (average of IMDb + TMDB)
            imdb_rating = row.get('avg_rating', 6.0)
            tmdb_rating = row.get('tmdb_rating', imdb_rating)

            if pd.isna(tmdb_rating):
                combined_rating = imdb_rating
            else:
                combined_rating = (imdb_rating + tmdb_rating) / 2.0

            num_votes = row['num_votes']

            # Calculate Option C score
            scores[idx] = self._calculate_option_c_score(
                combined_rating,
                num_votes,
                evening_type
            )

        return scores

    def _calculate_option_c_score(
        self,
        combined_rating: float,
        num_votes: int,
        evening_type: str
    ) -> float:
        """
        Linear quality scoring with 6.0 floor and smooth confidence curve.

        Candidates already filtered at 6.0, so this spreads them 0.0-1.0:
        - 6.0 rating → 0.0 quality score (minimum acceptable)
        - 7.0 rating → 0.25 quality score
        - 8.0 rating → 0.50 quality score
        - 9.0 rating → 0.75 quality score
        - 10.0 rating → 1.0 quality score

        Returns score between 0.0 and 1.0
        """
        # Step 1: Linear quality (6.0 → 0.0, 10.0 → 1.0)
        base_quality = (combined_rating - 6.0) / 4.0
        base_quality = max(0.0, min(1.0, base_quality))

        # Step 2: Smooth confidence multiplier (log-scale)
        confidence = self._calculate_vote_confidence(num_votes)

        # Step 3: Combine (no evening modifier - quality is quality)
        quality_score = base_quality * confidence

        return quality_score

    def _get_era_scores(self, candidates: pd.DataFrame, era: str) -> np.ndarray:
        """
        Calculate era favorability scores using TimePeriodFilter.

        Movies in preferred era get 1.0, outside get decay based on distance.
        This method is kept for backward compatibility but now uses TimePeriodFilter.
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            scores[idx] = TimePeriodFilter.calculate_era_score(row['year'], era)

        return scores

    def _get_keyword_scores(self, candidates: pd.DataFrame, selected_keywords: List[str]) -> np.ndarray:
        """
        Calculate keyword matching scores.

        Checks both TMDB keywords and descriptions.
        """
        # Filter out metadata keywords that shouldn't affect recommendations
        metadata_keywords = {
            'woman director', 'female director', 'male director',
            'directorial debut', 'independent film', 'film debut'
        }

        filtered_keywords = [kw for kw in selected_keywords if kw.lower() not in metadata_keywords]

        if not filtered_keywords:
            return np.zeros(len(candidates))

        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_dict = row.to_dict()
            scores[idx] = calculate_keyword_match_score(movie_dict, filtered_keywords)

        return scores

    def _get_keyword_scores_enhanced(
        self,
        candidates: pd.DataFrame,
        selected_keywords: Dict[str, Any],
        pacing_pref: Optional[str]
    ) -> np.ndarray:
        """
        Calculate enhanced keyword scores with age, general, adaptation, and pacing.

        Breakdown:
        - Age (35% of keyword score)
        - General keywords (40%)
        - Pacing (15%)
        - Adaptation (10%)
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            components = []

            # Age component (35%)
            if selected_keywords.get('age'):
                age_score = calculate_age_score(row['year'], selected_keywords['age'])
                components.append(age_score * 0.35)

            # General keywords component (40%)
            if selected_keywords.get('general'):
                general_score = self._match_general_keywords(row, selected_keywords['general'])
                components.append(general_score * 0.40)

            # Pacing component (15%)
            if selected_keywords.get('pacing'):
                # Extract pacing keyword value
                pacing_kw = selected_keywords['pacing']
                if isinstance(pacing_kw, list) and len(pacing_kw) > 0:
                    pacing_kw = pacing_kw[0]
                pacing_score = calculate_pacing_match(row, pacing_kw)
                components.append(pacing_score * 0.15)

            # Adaptation component (10%)
            if selected_keywords.get('adaptation'):
                adaptation_score = 1.0 if row.get('is_book_adaptation', False) else 0.0
                components.append(adaptation_score * 0.10)

            # Average components (only if any keywords selected)
            if components:
                scores[idx] = sum(components)

        return scores

    def _match_general_keywords(self, movie_row: pd.Series, user_keywords: List[str]) -> float:
        """
        Match movie against user's general keyword selections.
        Multi-level matching: exact, partial, description.
        """
        if not user_keywords:
            return 0.0

        movie_keywords = movie_row.get('keywords', [])
        description = movie_row.get('description', '')

        # Check if movie_keywords is empty (handle both lists and numpy arrays)
        has_keywords = False
        if isinstance(movie_keywords, (list, np.ndarray)):
            has_keywords = len(movie_keywords) > 0

        if not has_keywords and not description:
            return 0.0

        match_scores = []

        for user_kw in user_keywords:
            # Level 1: Exact match in TMDb keywords
            if has_keywords and isinstance(movie_keywords, (list, np.ndarray)):
                exact_match = any(
                    user_kw.lower() == str(movie_kw).lower()
                    for movie_kw in movie_keywords
                )

                if exact_match:
                    match_scores.append(1.0)
                    continue

                # Level 2: Partial match in TMDb keywords
                partial_match = any(
                    user_kw.lower() in str(movie_kw).lower() or str(movie_kw).lower() in user_kw.lower()
                    for movie_kw in movie_keywords
                )

                if partial_match:
                    match_scores.append(0.7)
                    continue

            # Level 3: Description match
            if description:
                user_kw_formatted = user_kw.replace('_', ' ').lower()
                if user_kw_formatted in description.lower():
                    match_scores.append(0.4)
                    continue

            # No match
            match_scores.append(0.0)

        # Average match score
        return sum(match_scores) / len(user_keywords) if match_scores else 0.0

    def _get_quality_scores_simple(self, candidates: pd.DataFrame) -> np.ndarray:
        """
        Simplified quality scoring without evening modifiers.
        Uses Option C formula: base_quality * confidence
        """
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            # Base quality (rating normalized to 0-1)
            combined_rating = row['avg_rating']
            base_quality = (combined_rating - 6.0) / 4.0
            base_quality = max(0.0, min(1.0, base_quality))

            # Confidence based on vote count
            num_votes = row['num_votes']
            confidence = self._calculate_vote_confidence(num_votes)

            scores[idx] = base_quality * confidence

        return scores

    def _get_session_scores_enhanced(
        self,
        candidates: pd.DataFrame,
        genres: List[str],
        session_positive_movies: List[int],
        session_history: List[Dict],
        pacing_pref: Optional[str]
    ) -> np.ndarray:
        """
        Enhanced session scoring with pacing awareness.
        """
        if not session_positive_movies:
            return np.ones(len(candidates)) * 0.5

        # Get liked movies data
        liked_movies = [
            item for item in session_history
            if item.get('action') in ['right', 'up']
        ]

        if not liked_movies:
            return np.ones(len(candidates)) * 0.5

        # Extract patterns from liked movies
        liked_genres = set()
        liked_years = []
        liked_pacing = []

        for item in liked_movies:
            movie_id = item['movie_id']
            movie_data = self.movies[self.movies['movie_id'] == movie_id]

            if len(movie_data) > 0:
                movie = movie_data.iloc[0]
                if isinstance(movie.get('genres'), (list, np.ndarray)):
                    liked_genres.update(movie['genres'])
                liked_years.append(movie['year'])

                if 'pacing_score' in item:
                    liked_pacing.append(item['pacing_score'])

        avg_year = np.mean(liked_years) if liked_years else 2015

        # Score candidates
        scores = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            score = 0.0

            # Genre overlap (40%)
            if liked_genres:
                movie_genres = set(row['genres']) if isinstance(row.get('genres'), (list, np.ndarray)) else set()
                overlap = len(movie_genres & liked_genres) / len(liked_genres)
                score += overlap * 0.4

            # Year similarity (30%)
            year_diff = abs(row['year'] - avg_year)
            year_score = max(0, 1.0 - (year_diff / 20))
            score += year_score * 0.3

            # Pacing similarity (30%)
            if pacing_pref:
                # Use inferred pacing preference
                movie_pacing = row.get('pacing_score', 0.5)

                if pacing_pref == 'fast' and movie_pacing > 0.7:
                    score += 0.3
                elif pacing_pref == 'slow' and movie_pacing < 0.4:
                    score += 0.3
                else:
                    score += 0.1
            else:
                score += 0.15

            scores[idx] = score

        return scores

    def _calculate_popularity_boosts(
        self,
        candidates: pd.DataFrame,
        popularity_mode: str
    ) -> np.ndarray:
        """
        Calculate popularity boost multipliers.
        """
        boosts = np.zeros(len(candidates))

        for idx, (_, row) in enumerate(candidates.iterrows()):
            boosts[idx] = calculate_popularity_boost(row['num_votes'], popularity_mode)

        return boosts

    def suggest_keywords(self, genres: List[str], num_keywords: int = 8) -> List[str]:
        """
        Suggest contextual keywords based on selected genres.

        Args:
            genres: List of 1-2 selected genres
            num_keywords: Number of keywords to suggest

        Returns:
            List of keyword strings
        """
        if not self.keyword_analyzer:
            return []

        return self.keyword_analyzer.suggest_keywords(genres, num_keywords)

    def _get_cf_score_single(self, user_id: int, movie_id: int) -> float:
        """
        Get Collaborative Filtering score for a single movie.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            CF score normalized to [0, 1]
        """
        try:
            if not hasattr(self.base_engine, 'cf_model') or self.base_engine.cf_model is None:
                return 0.5  # Neutral score if CF model unavailable
            
            # Predict using CF model (returns score in [-1, 1])
            cf_score = self.base_engine.cf_model.predict(user_id, movie_id)
            
            # Normalize from [-1, 1] to [0, 1]
            normalized_score = (cf_score + 1) / 2.0
            return max(0.0, min(1.0, normalized_score))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"CF prediction failed for user {user_id}, movie {movie_id}: {e}")
            return 0.5
    
    def _get_graph_score_single(self, movie_id: int, session_positive_movies: List[int]) -> float:
        """
        Get Graph co-occurrence score for a single movie.
        
        Args:
            movie_id: Movie ID
            session_positive_movies: List of movie IDs user liked in session
            
        Returns:
            Graph score in [0, 1]
        """
        if not session_positive_movies:
            return 0.0
        
        try:
            if not hasattr(self.base_engine, 'graph') or self.base_engine.graph is None:
                return 0.0
            
            # Get max similarity from last 5 positive movies
            max_similarity = 0.0
            for liked_id in session_positive_movies[-5:]:
                try:
                    neighbors = dict(self.base_engine.graph.get_neighbors(liked_id, top_k=100))
                    similarity = neighbors.get(movie_id, 0.0)
                    max_similarity = max(max_similarity, similarity)
                except:
                    continue
            
            return min(1.0, max_similarity)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Graph score failed for movie {movie_id}: {e}")
            return 0.0
    
    def _get_content_score_single(
        self,
        movie_id: int,
        session_positive_movies: List[int],
        session_negative_movies: List[int]
    ) -> float:
        """
        Get Content Similarity score for a single movie.
        
        Args:
            movie_id: Movie ID
            session_positive_movies: List of movie IDs user liked
            session_negative_movies: List of movie IDs user rejected
            
        Returns:
            Content score in [0, 1]
        """
        if self.content_similarity is None:
            return 0.0
        
        if not session_positive_movies and not session_negative_movies:
            return 0.0
        
        try:
            positive_signal = 0.0
            if session_positive_movies:
                # Get max similarity to positive movies
                for liked_id in session_positive_movies[-5:]:
                    try:
                        sim = self.content_similarity.get_similarity(movie_id, liked_id)
                        positive_signal = max(positive_signal, sim)
                    except:
                        continue
            
            negative_signal = 0.0
            if session_negative_movies:
                # Get max similarity to negative movies (penalty)
                for rejected_id in session_negative_movies[-5:]:
                    try:
                        sim = self.content_similarity.get_similarity(movie_id, rejected_id)
                        negative_signal = max(negative_signal, sim)
                    except:
                        continue
            
            # Combine: positive signal - (negative signal * 0.5)
            # Negative penalty is weaker to avoid over-penalization
            content_score = positive_signal - (negative_signal * 0.5)
            return max(0.0, min(1.0, content_score))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Content similarity failed for movie {movie_id}: {e}")
            return 0.0

    def _score_candidates_new(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        genres: List[str],
        era: str,
        source_material: str,
        themes: List[str],
        session_positive_movies: List[int],
        session_history: List[Dict],
        db_history: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Score candidates with hybrid ML system.
        
        IMPORTANT: candidates DataFrame already passed hard filters.
        Feedback adjustments only affect ranking within this pool.
        
        Scoring Formula:
        - Explicit Components (70%):
          + Genre Match × 0.30
          + Era Match × 0.15
          + Keyword Match × 0.20
          + Quality × 0.05
        - ML Components (25%):
          + CF Score × 0.15
          + Graph Score × 0.10
        - Content Similarity (5%):
          + Content Score × 0.05 (if available, else 0.0)
        - Feedback Adjustment (±10%):
          + User-specific feedback ±10%
        - Selection-Movie Compatibility (±5%):
          + User-Context Compatibility × 0.05 (per-user, context-specific)
        - Global Context Fit (±5%):
          + Global Context-Movie Fit × 0.05 (across all users, context-specific)
        
        Final Score = Base Score + Feedback Adjustment + Compatibility Adjustment + Global Fit Adjustment
        """
        num_candidates = len(candidates)
        scores = np.zeros(num_candidates)
        
        # Extract negative movies from session history
        session_negative = [
            item['movie_id'] for item in session_history
            if item.get('action') in ['left', 'no']
        ]
        
        for idx, (_, row) in enumerate(candidates.iterrows()):
            movie_id = row['movie_id']
            
            # EXPLICIT COMPONENTS (70%)
            # 1. Genre matching (30%)
            movie_genres = set(row['genres']) if isinstance(row['genres'], (list, np.ndarray)) else set()
            genre_set = set(genres)
            genre_overlap = len(movie_genres & genre_set) / len(genre_set) if genre_set else 0
            genre_score = genre_overlap * 0.30
            
            # 2. Era matching (15%)
            era_score = TimePeriodFilter.calculate_era_score(row['year'], era) * 0.15
            
            # 3. Theme keywords matching (20%)
            movie_keywords = row.get('keywords', [])
            if themes:
                theme_matches = 0
                if isinstance(movie_keywords, (list, np.ndarray)):
                    movie_kw_lower = {str(kw).lower() for kw in movie_keywords}
                    for theme in themes:
                        if theme.lower() in movie_kw_lower:
                            theme_matches += 1
                theme_score = (theme_matches / len(themes)) * 0.20 if themes else 0
            else:
                theme_score = 0.20  # Full score if no themes specified
            
            # 4. Quality score (5%)
            quality_score = self._calculate_simple_quality(row) * 0.05
            
            explicit_score = genre_score + era_score + theme_score + quality_score
            
            # ML COMPONENTS (25%)
            cf_score = self._get_cf_score_single(user_id, movie_id) * 0.15
            graph_score = self._get_graph_score_single(movie_id, session_positive_movies) * 0.10
            ml_score = cf_score + graph_score
            
            # CONTENT SIMILARITY (5%)
            content_score = self._get_content_score_single(
                movie_id, session_positive_movies, session_negative
            ) * 0.05
            
            # BASE SCORE (100%)
            base_score = explicit_score + ml_score + content_score
            
            # FEEDBACK ADJUSTMENT (±10%)
            # Get feedback adjustment from persistent learner (context-scoped to current genres+era)
            feedback_adjustment = 0.0
            if self.feedback_learner and db_history is not None:
                try:
                    graph_model = getattr(self.base_engine, 'graph', None)
                    feedback_adjustment = self.feedback_learner.get_adjustment(
                        movie_id=movie_id,
                        user_id=user_id,
                        session_id=None,
                        user_history=db_history,  # Use separate DB history (not merged with session)
                        content_similarity_model=self.content_similarity,
                        graph_model=graph_model,
                        current_genres=genres,  # Pass current context for filtering
                        current_era=era,  # Pass current era for filtering
                        min_context_observations=5  # Need at least 5 interactions in this context
                    )
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Feedback adjustment failed for movie {movie_id}: {e}")
                    feedback_adjustment = 0.0
            
            # SELECTION-MOVIE COMPATIBILITY (±5%)
            compatibility_adjustment = 0.0
            if self.feedback_learner:
                try:
                    compatibility_score = self.feedback_learner.get_compatibility_score(
                        user_id=user_id,
                        movie_id=movie_id,
                        genres=genres,
                        era=era,
                        themes=themes
                    )
                    # Map [-1.0, +1.0] to [-0.05, +0.05] (5% weight)
                    compatibility_adjustment = compatibility_score * 0.05
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Compatibility score failed for movie {movie_id}: {e}")
                    compatibility_adjustment = 0.0
            
            # GLOBAL CONTEXT FIT (±5%)
            # Learn from aggregated feedback across all users for this context
            global_fit_adjustment = 0.0
            if self.feedback_learner:
                try:
                    global_fit_score = self.feedback_learner.get_global_context_fit_score(
                        movie_id=movie_id,
                        genres=genres,
                        era=era,
                        themes=themes,
                        min_observations=5,  # Need at least 5 observations to trust
                        alpha=2.0,  # Beta smoothing parameters
                        beta=2.0
                    )
                    # Map [-0.5, +0.5] to [-0.05, +0.05] (5% weight, max 10% of range)
                    # This bounds the adjustment safely
                    global_fit_adjustment = global_fit_score * 0.10
                    global_fit_adjustment = np.clip(global_fit_adjustment, -0.05, +0.05)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Global context fit score failed for movie {movie_id}: {e}")
                    global_fit_adjustment = 0.0
            
            # FINAL SCORE
            scores[idx] = base_score + feedback_adjustment + compatibility_adjustment + global_fit_adjustment
            scores[idx] = np.clip(scores[idx], 0.0, 1.0)  # Clamp to [0, 1]
        
        return scores

    def _calculate_simple_quality(self, movie_row: pd.Series) -> float:
        """Simple quality calculation based on rating and votes"""
        rating = movie_row.get('avg_rating', 6.0)
        num_votes = movie_row.get('num_votes', 0)

        # Normalize rating (6.0-10.0 → 0.0-1.0)
        rating_score = (rating - 6.0) / 4.0
        rating_score = max(0.0, min(1.0, rating_score))

        # Vote confidence
        confidence = self._calculate_vote_confidence(num_votes)

        return rating_score * confidence


def load_smart_system(
    models_dir: Path,
    movies_path: Path,
    keyword_db_path: Optional[Path] = None,
    feedback_db_path: Optional[Path] = None
) -> SmartRecommendationEngine:
    """
    Load complete smart recommendation system.

    Args:
        models_dir: Directory with CF and graph models
        movies_path: Path to movies.parquet
        keyword_db_path: Optional path to keyword_database.pkl
        feedback_db_path: Optional path to feedback database (default: output/feedback.db)

    Returns:
        SmartRecommendationEngine instance
    """
    from src.recommendation.recommendation_engine import load_system as load_base
    from src.models.content_similarity import ContentSimilarity

    # Load base engine (CF + graph)
    base_engine = load_base(models_dir, movies_path)

    # Load keyword analyzer if available
    keyword_analyzer = None
    if keyword_db_path and keyword_db_path.exists():
        keyword_analyzer = KeywordAnalyzer.load(keyword_db_path)

    # Load Content Similarity model if available (optional)
    # Auto-train on first run if missing
    content_similarity = None
    content_sim_path = models_dir / 'content_similarity.pkl'
    logger = logging.getLogger(__name__)
    if content_sim_path.exists():
        try:
            content_similarity = ContentSimilarity.load(models_dir)
            logger.info("Content Similarity model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ContentSimilarity model: {e}")
    else:
        # Auto-train content similarity if movies.parquet exists (first-run cache)
        if movies_path.exists():
            logger.info("Content Similarity model not found -> training now (first-run cache)...")
            try:
                from src.models.content_similarity import train_content_similarity
                train_content_similarity(movies_path, models_dir)
                logger.info("✓ ContentSimilarity trained and saved")
                # Now try to load it
                try:
                    content_similarity = ContentSimilarity.load(models_dir)
                    logger.info("Content Similarity model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load newly trained ContentSimilarity model: {e}")
            except Exception as e:
                logger.warning(f"Content similarity training failed: {e}")
                logger.warning("Continuing without content similarity (optional component)")
        else:
            logger.info("Content Similarity model not found and movies.parquet missing - continuing without it")

    # Set default feedback database path if not provided
    if feedback_db_path is None:
        # Default to output/feedback.db relative to models_dir
        feedback_db_path = models_dir.parent / 'feedback.db'

    # Create smart engine
    smart_engine = SmartRecommendationEngine(
        base_engine=base_engine,
        keyword_analyzer=keyword_analyzer,
        content_similarity=content_similarity,
        feedback_db_path=feedback_db_path
    )

    return smart_engine


if __name__ == '__main__':
    """Quick test of smart engine."""
    models_dir = Path('output/models')
    movies_path = Path('output/processed/movies.parquet')
    keyword_db_path = Path('output/models/keyword_database.pkl')

    print("Loading smart recommendation system...")
    engine = load_smart_system(models_dir, movies_path, keyword_db_path)

    print("\nTesting recommendations:")

    # Test with keywords
    selected_keywords = {
        'age': 'recent',
        'general': ['espionage', 'heist'],
        'adaptation': False,
        'pacing': ['fast_paced']
    }

    recommendations = engine.recommend(
        user_id=0,
        genres=['action', 'thriller'],
        selected_keywords=selected_keywords,
        session_history=[],
        top_k=10
    )

    print(f"\nTop 10 recommendations (with keywords):")
    for rank, movie_id in enumerate(recommendations, 1):
        movie = engine.movies[engine.movies['movie_id'] == movie_id].iloc[0]
        print(f"{rank}. {movie['title']} ({movie['year']}) - {movie['avg_rating']:.1f}/10")
