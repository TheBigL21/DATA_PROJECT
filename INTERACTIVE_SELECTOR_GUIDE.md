# Interactive Movie Selector - Complete Guide

## Overview

The Interactive Movie Selector is an adaptive recommendation system that learns from user feedback in real-time. Instead of showing a static list of 10 movies, it presents **one movie at a time** and evolves the candidate pool based on your choices.

## How It Works

### 1. Initial Setup (4 Questions)

You answer 4 questions to establish context:

1. **Genres** (1-2): action, thriller, comedy, drama, horror, sci-fi, romance, adventure
2. **Era**: fresh (2020+), modern (2015+), timeless (2000+), old_school (classic)
3. **Source Material**: book adaptations or any
4. **Themes** (optional): heist, espionage, revenge, family, coming-of-age, etc.

### 2. Interactive Selection Loop

The system shows you **one movie at a time** with:
- Title and year
- Rating
- Genres
- Keywords

You have 3 choices:

| Input | Action | Meaning | System Response |
|-------|--------|---------|-----------------|
| `0` or `Enter` | **YES** | Like it, show similar | Adds similar movies to pool, removes dissimilar ones |
| `1` | **NO** | Not interested, show different | Removes similar movies, adds diverse ones |
| `2` | **FINAL** | Choose this movie! | Shows full details and ends session |

### 3. Adaptive Learning

The system learns at **three levels**:

#### A. Session-Level Learning (Primary)
- Tracks which features you like/dislike in current session
- Updates feature weights after each choice
- Example: If you say "yes" to action movies, it boosts `genre_action` weight

#### B. Sequence-Level Learning (Cross-User)
- Learns which movie pairs work well together
- Example: If many users like Movie B after Movie A, it boosts that transition
- Stored persistently across all sessions

#### C. Global Learning (Cross-User)
- Tracks movie success rates by context
- Example: If Movie X gets "no" from 80% of users asking for "action + modern", it gets penalized
- Helps avoid bad recommendations for future users

### 4. Dynamic Candidate Pool

Unlike static top-10 lists, the pool **evolves**:

**Initial State:**
- 80-100 movies from SmartEngine matching your context

**After "YES" (0 or Enter):**
- Finds movies similar to liked movie (similarity > 0.6)
- Adds 15 most similar movies to pool
- Removes dissimilar movies (similarity < 0.3)
- Pool shifts toward liked features

**After "NO" (1):**
- Removes movies similar to rejected movie (similarity > 0.5)
- Adds 15 diverse movies from catalog
- Pool shifts away from rejected features

**Maintenance:**
- Pool size maintained around 80 movies
- All movies respect original context constraints (genres, era, source)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER ANSWERS 4 QUESTIONS                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           SmartEngine.recommend() → 100 candidates          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AdaptivePoolManager (80 active)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          InteractiveSelector picks best movie               │
│                                                             │
│  Score = 0.45 * SessionLearner                              │
│        + 0.25 * SequenceLearner                             │
│        + 0.30 * GlobalLearner                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    SHOW MOVIE → GET INPUT                   │
│                    (0=yes, 1=no, 2=final)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS FEEDBACK                         │
│                                                             │
│  1. Update SessionLearner (feature weights)                 │
│  2. Update SequenceLearner (movie A → B transitions)        │
│  3. Update GlobalLearner (movie quality stats)              │
│  4. Adapt pool (expand similar or diverse)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                   LOOP UNTIL "FINAL"
```

## Components

### 1. FeatureEncoder

Converts movies to binary feature vectors:

**Features (150+ dimensions):**
- Genre multi-hot (18 genres)
- Keyword multi-hot (100 top keywords)
- Era bins (classic, old, modern, recent)
- Pacing bins (slow, medium, fast)
- Quality bins (low, medium, high)

**Similarity:**
- Cosine similarity on feature vectors
- Range: 0.0 (completely different) to 1.0 (identical)

### 2. SessionLearner

Learns feature preferences within current session.

**Algorithm:**
- Maintains weight vector over all features
- Updates after each action:
  - `yes`: `weights += 0.2 * feature_vector`
  - `no`: `weights -= 0.2 * feature_vector`
  - `final`: `weights += 0.4 * feature_vector`

**Scoring:**
```python
score = dot(weights, movie_features)
```

### 3. SequenceLearner

Learns pairwise movie compatibility.

**Data Structure:**
```python
transitions[(movie_A, movie_B)] = [count, success_count, total_satisfaction]
```

**Compatibility Score:**
```python
if count >= 2:
    compatibility = success_count / count  # Empirical success rate
else:
    compatibility = 0.5  # Neutral
```

### 4. GlobalLearner

Tracks movie quality across all users.

**Data Structures:**
- `context_stats[context_key][movie_id] = [count, success_count, satisfaction]`
- `global_stats[movie_id] = [count, success_count, satisfaction]`

**Penalty/Boost:**
```python
if count >= 5:  # Context-specific
    success_rate = success_count / count
    penalty = (success_rate - 0.5)  # Range: -0.5 to +0.5

elif global_count >= 10:  # Fallback to global
    success_rate = global_success / global_count
    penalty = (success_rate - 0.5) * 0.5  # Weaker signal
```

**Example:**
- Movie with 20% success rate → penalty = -0.3
- Movie with 80% success rate → boost = +0.3

### 5. AdaptivePoolManager

Manages evolving candidate pool.

**Methods:**

- `expand_similar(reference_movie, threshold=0.6, add=15)`: Add similar movies after "yes"
- `expand_diverse(avoid_movie, threshold=0.5, add=15)`: Add diverse movies after "no"
- `maintain_size()`: Keep pool around 80 movies
- `get_available()`: Return unshown movies from pool

**Constraints:**
- All added movies must match original context (genres, era, source)
- Built from full catalog respecting user's initial answers

### 6. InteractiveSelector

Orchestrates selection and learning.

**Scoring Formula:**
```python
for each candidate:
    session_score = SessionLearner.score(movie)
    sequence_score = SequenceLearner.compatibility(last_movie, movie)
    global_penalty = GlobalLearner.penalty(context, movie)

    total_score = (
        0.45 * session_score +
        0.25 * sequence_score +
        0.30 * global_penalty
    )

best_movie = argmax(total_score)
```

## Usage

### Running the CLI

```bash
cd /Users/louismegow/DATA_PROJECT
python -m src.recommendation.interactive_cli
```

### Example Session

```
MOVIE RECOMMENDATION - INTERACTIVE MODE

1. What genres are you interested in? (select 1-2)
   Enter genres: action, thriller

2. What time period?
   Enter era: modern

3. Source material preference?
   Enter preference: any

4. Any specific themes?
   Enter themes: heist, espionage

✓ Context collected
Generating initial movie candidates...
✓ Generated 100 initial candidates

------------------------------------------------------------
Movie #1
------------------------------------------------------------
Inception (2010)
Rating: 8.8/10
Genres: action, thriller, sci-fi
Keywords: heist, dream, mind-bending, subconscious, complex
------------------------------------------------------------

Your choice:
  0 or Enter → Yes (like it, show similar)
  1 → No (not interested, show different)
  2 → FINAL (choose this movie!)

Choice: 0
→ Yes, show similar movies

------------------------------------------------------------
Movie #2
------------------------------------------------------------
The Prestige (2006)
Rating: 8.5/10
Genres: drama, thriller, mystery
Keywords: magic, obsession, revenge, rivalry, twist-ending
------------------------------------------------------------

Choice: 1
→ No, show different movies

------------------------------------------------------------
Movie #3
------------------------------------------------------------
Heat (1995)
Rating: 8.3/10
Genres: action, thriller, crime
Keywords: heist, cat-and-mouse, professional, los-angeles
------------------------------------------------------------

Choice: 2
→ FINAL CHOICE!

================================================================================
YOUR MOVIE CHOICE:
================================================================================

Title: Heat
Year: 1995
Rating: 8.3/10 (672,891 votes)
Genres: action, thriller, crime
Runtime: 170 minutes
Director: Michael Mann
Cast: Al Pacino, Robert De Niro, Val Kilmer, Jon Voight, Tom Sizemore

Keywords: heist, cat-and-mouse, professional, los-angeles, shootout, ...

Description:
Hunters and their prey--Neil and his professional criminal crew hunt to score
big money targets (banks, vaults, armored cars) and are, in turn, hunted by
Lt. Vincent Hanna and his team of cops in the Robbery/Homicide police division...

================================================================================
Enjoy your movie! 🎬
================================================================================

SESSION SUMMARY
Movies shown: 3

Learned preferences:
  👍 genre_action: +0.40
  👍 keyword_heist: +0.60
  👎 keyword_magic: -0.20
  👍 era_modern: +0.20
  👍 pacing_fast: +0.40
```

## Tuning Parameters

### Satisfaction Mapping

Current mapping (can adjust in code):

```python
action_to_satisfaction = {
    'yes': 0.3,    # Like but not watching now
    'no': 0.0,     # Reject
    'final': 1.0   # Accept
}
```

**Adjustment Strategy:**
- Increase `'yes'` → 0.4-0.5 if you want stronger positive signal
- Add intermediate levels if needed (e.g., 'maybe': 0.15)

### Learning Rates

`SessionLearner.learning_rate = 0.2`
- Higher (0.3-0.4): Faster adaptation, more volatile
- Lower (0.1-0.15): Slower adaptation, more stable

### Pool Thresholds

`expand_similar(similarity_threshold=0.6)`
- Higher (0.7-0.8): Stricter similarity, fewer additions
- Lower (0.5): More lenient, faster pool growth

`expand_diverse(diversity_threshold=0.5)`
- Higher (0.6): Remove more similar movies
- Lower (0.4): Keep more borderline movies

### Scoring Weights

```python
w_session = 0.45   # Session learning (current preferences)
w_sequence = 0.25  # Sequence compatibility (what works together)
w_global = 0.30    # Global quality (crowd wisdom)
```

**Personalization:**
- Increase `w_session` → 0.5-0.6 for more responsive to current user
- Increase `w_global` → 0.4 for more crowd-validated recommendations
- Increase `w_sequence` → 0.3 for smoother transitions between movies

### Global Learning Thresholds

```python
# Context-specific minimum observations
if count >= 5:
    use context-specific stats

# Global minimum observations
elif global_count >= 10:
    use global stats (weaker)
```

**Adjustment:**
- Lower thresholds (3, 5): Faster learning, less reliable
- Higher thresholds (10, 20): Slower learning, more reliable

## Persistence

### What Gets Saved

**Per Session:**
- Feature weights (SessionLearner) - discarded after session
- Feedback events - logged for analysis

**Across Sessions:**
- Sequence transitions (SequenceLearner) - saved to `output/interactive_learning/learning_models.pkl`
- Global movie stats (GlobalLearner) - saved to same file

### Storage Format

```python
# Saved as pickle
{
    'sequence': {
        'transitions': {
            '123_456': [10, 7, 5.4],  # movie_A_id_movie_B_id: [count, success, satisfaction]
            ...
        }
    },
    'global': {
        'context_movie_stats': {
            'action,thriller|modern|any': {
                123: [15, 10, 8.5],  # movie_id: [count, success, satisfaction]
                ...
            },
            ...
        },
        'global_movie_stats': {
            123: [50, 35, 22.3],  # movie_id: [count, success, satisfaction]
            ...
        }
    }
}
```

## Integration with Existing Pipeline

The InteractiveSelector **plugs into** your existing SmartEngine:

```python
from recommendation.smart_engine import load_smart_system
from recommendation.interactive_selector import (
    FeatureEncoder,
    InteractiveSelector,
    ...
)

# Load your existing system
smart_engine = load_smart_system(models_dir, movies_path, keyword_db_path)

# Add interactive layer
encoder = FeatureEncoder(smart_engine.movies)
selector = InteractiveSelector(encoder, ...)

# Use SmartEngine for initial candidates
initial_candidates = smart_engine.recommend(
    user_id, genres, era, source_material, themes, top_k=100
)

# Interactive selector refines from there
pool_manager = AdaptivePoolManager(..., initial_candidates, ...)
```

**Clean Separation:**
- SmartEngine: Candidate generation (uses CF, graph, keywords)
- InteractiveSelector: Adaptive re-ranking and learning

## Behavioral Examples

### Example 1: User Likes Heist Movies

**Session:**
1. Show "Ocean's Eleven" → User says **yes** (0)
2. System learns: `keyword_heist += 0.2`, `genre_thriller += 0.2`
3. Pool expands to similar movies: adds "The Italian Job", "Baby Driver"
4. Show "The Italian Job" → User says **yes** (0)
5. Weights strengthen: `keyword_heist += 0.4`, `keyword_car_chase += 0.2`
6. Show "Baby Driver" → User says **final** (2)

**Global Learning:**
- Transition "Ocean's Eleven" → "The Italian Job": success recorded
- Transition "The Italian Job" → "Baby Driver": success recorded
- Future users in similar contexts get smoother heist movie progression

### Example 2: Movie Consistently Rejected

**Across Multiple Users:**
- Movie "Generic Action 5" shown to 20 users in context "action + modern"
- 18 users say "no", 2 say "yes"
- Success rate: 2/20 = 10%
- Penalty: (0.1 - 0.5) = **-0.4**

**Impact:**
- Movie "Generic Action 5" gets strong negative penalty in that context
- Future users unlikely to see it
- System learns this movie doesn't fit the context well

### Example 3: Repeated "No" Triggers Exploration

**Session:**
1. User says "no" to Movie A
2. User says "no" to Movie B
3. User says "no" to Movie C
4. After 3 consecutive "no", system detects pattern
5. Next selection heavily weights diversity
6. Shows movie from different feature cluster
7. User finds something they like

## Debugging

### Check Learned Features

Session summary prints top learned features:

```
Learned preferences:
  👍 genre_action: +0.40
  👍 keyword_heist: +0.60
  👎 keyword_magic: -0.20
```

### Inspect Active Features

```python
active_features = encoder.get_active_features(movie_id)
print(active_features)
# ['genre_action', 'genre_thriller', 'keyword_heist', 'era_modern', ...]
```

### Check Similarity

```python
similarity = encoder.compute_similarity(movie_a_id, movie_b_id)
print(f"Similarity: {similarity:.2f}")
# Similarity: 0.78
```

### View Global Stats

```python
worst_movies = global_learner.get_worst_movies(context, top_k=10)
for movie_id, success_rate in worst_movies:
    print(f"Movie {movie_id}: {success_rate:.1%} success")
```

## Future Enhancements

**Potential additions:**

1. **Multi-armed bandit**: Replace scoring with UCB or Thompson Sampling for better exploration/exploitation
2. **Deep features**: Use movie embeddings from neural CF instead of binary features
3. **User clustering**: Group similar users, transfer learning between them
4. **Temporal decay**: Weight recent feedback more than old feedback
5. **Confidence intervals**: Show uncertainty estimates to user
6. **A/B testing**: Compare different scoring weight configurations

## Summary

**Key Innovation:** Dynamic candidate pool that evolves based on your choices, not static re-ranking.

**Learning Hierarchy:**
1. **Session** (fastest): Your current preferences
2. **Sequence** (medium): What movie pairs work together
3. **Global** (slowest): Crowd-validated movie quality

**User Experience:**
- Simple input: 0 (yes), 1 (no), 2 (final)
- Progressive refinement: Each choice makes next recommendation better
- Full control: System adapts to your exact preferences

**Production-Ready:**
- Persistent learning across sessions
- Scalable to many users
- Integrates cleanly with existing SmartEngine
- ~600 lines of clean, documented code
