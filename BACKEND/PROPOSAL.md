# Project Proposal: Movie Recommendation System


## Problem Statement

Users spend significant time browsing streaming platforms (Netflix, Amazon Prime, Disney+) without finding suitable content. Studies show that the average user spends 18 minutes searching before selecting a movie, and 25% of users abandon the search entirely.

Existing recommendation systems optimize for platform engagement and profit maximization rather than user satisfaction. These systems often:
- Promote content with high profit margins
- Recommend trending content regardless of user preferences
- Fail to respect explicit user intent (genre, mood, context)
- Provide opaque recommendations without explanation

There is a need for a transparent, user-centric recommendation system that prioritizes matching user preferences over platform business objectives.

---

## Proposed Solution

A hybrid machine learning movie recommendation system that respects explicit user preferences while providing personalized suggestions. The system will:

1. Allow users to specify their viewing context (date night, family night, solo viewing)
2. Respect user-selected genres with 100% match rate
3. Provide rich metadata (posters, descriptions, keywords) for informed decisions
4. Combine multiple ML approaches for robust recommendations
5. Adapt to user feedback within the session

---

## Technical Approach

### Data Sources

**IMDb Dataset (Primary)**
- Source: https://datasets.imdbws.com/
- Content: 45,000+ quality movies (filtered from millions)
- Filter criteria: ≥1000 votes, valid ratings, complete metadata
- Features: Titles, years, genres, runtime, ratings, cast, crew

**TMDb API (Enrichment)**
- Source: https://www.themoviedb.org/
- Content: Rich metadata for each movie
- Features: High-quality posters, plot descriptions, thematic keywords, budget/revenue data
- Coverage: 99.3% of filtered IMDb dataset

### Data Pipeline

**Stage 1: Data Cleaning**
- Filter IMDb dataset from millions to ~45K quality movies
- Criteria: Movies only, non-adult, 30-400 min runtime, ≥1000 votes
- Result: High-quality, reliable dataset

**Stage 2: TMDb Enrichment**
- Match IMDb IDs to TMDb IDs via API
- Fetch metadata for each movie (rate-limited to 10 req/sec)
- Duration: 2-4 hours for full dataset
- Result: 99.3% coverage with rich metadata

**Stage 3: Feature Engineering**
- Extract country of origin from alternative titles
- Generate genre-specific keyword databases
- Create composite ratings (IMDb + TMDb weighted average)
- Assign integer movie IDs for efficient ML operations

**Stage 4: Synthetic User Generation**
- Generate 1000 synthetic users with realistic preference patterns
- Simulate user-movie interactions (swipe actions)
- Purpose: Train collaborative filtering model (cold-start solution)

### Machine Learning Models

**Model 1: Collaborative Filtering (ALS)**
- Algorithm: Alternating Least Squares for implicit feedback
- Approach: Matrix factorization (R ≈ U × M^T)
- Parameters: 50 latent dimensions, λ=0.01 regularization, 20 iterations
- Purpose: Capture "users like you also liked" patterns
- Advantage: Discovers non-obvious preferences

**Model 2: Co-occurrence Graph**
- Algorithm: Item-based collaborative filtering via graph structure
- Approach: Build graph where edges represent co-viewing frequency
- Structure: Sparse adjacency matrix (45K × 45K)
- Purpose: Recommend movies frequently enjoyed together
- Advantage: Session-based recommendations without user history

**Model 3: Content Similarity (TF-IDF)**
- Algorithm: Term Frequency-Inverse Document Frequency + Cosine Similarity
- Features: Genres (weighted), keywords (TF-IDF), directors (high weight)
- Purpose: Match movies by intrinsic characteristics
- Advantage: Works without interaction data, ensures thematic consistency

**Model 4: Keyword Database**
- Algorithm: Inverted index with frequency ranking
- Structure: Keyword → movie list mapping
- Purpose: Fast keyword-based filtering and suggestions
- Advantage: Enables user refinement with specific themes

### Recommendation Algorithm

**Multi-Stage Hybrid Approach:**

**Stage 1: Candidate Generation**
- Hard genre filter (must match user selection)
- Top-300 from collaborative filtering (genre-filtered)
- Top-100 from co-occurrence graph (session-based)
- Top-100 popular movies in genre (cold-start fallback)
- Result: 500-1000 candidates

**Stage 2: Composite Scoring**
```
score = 25% × genre_match          # Explicit user intent
      + 25% × CF_score             # Collaborative filtering
      + 15% × quality_score        # Rating + vote confidence
      + 15% × graph_score          # Co-occurrence
      + 10% × session_score        # Session similarity
      + 10% × other                # Era, keywords, region
```

**Stage 3: Re-ranking**
- Remove already-shown movies
- Apply diversity boost (avoid genre repetition)
- Return top-K (typically 10) recommendations

### System Architecture

**Backend (Python)**
- Framework: Flask REST API
- ML Libraries: NumPy, pandas, SciPy
- Data Storage: Parquet (compressed columnar format)
- Components:
  - Data pipeline (IMDb cleaning, TMDb fetching)
  - ML model training (CF, graph, content similarity)
  - Recommendation engine (smart_engine.py)
  - API server (api_smart.py)

**Frontend (React/TypeScript)**
- Framework: React 18 with TypeScript
- Build Tool: Vite (fast dev server, optimized builds)
- UI Library: shadcn/ui + Tailwind CSS
- Features:
  - Questionnaire interface
  - Swipeable movie cards (left/right/up)
  - Real-time keyword suggestions
  - Results display with rich metadata

**Communication**
- Protocol: REST API (HTTP)
- Format: JSON
- CORS: Enabled for cross-origin requests
- Endpoints: /api/recommend, /api/feedback, /api/questionnaire/*

---

## Expected Outcomes

### Quantitative Metrics

| Metric | Target | Justification |
|--------|--------|---------------|
| Genre Match Rate | 100% | All recommendations match user-selected genres |
| Average Quality | 7.5+/10 | Only high-quality movies recommended |
| TMDb Coverage | 95%+ | Most movies have rich metadata |
| Response Time | <500ms | Real-time recommendations |
| Dataset Size | 40K+ | Sufficient diversity for recommendations |

### Qualitative Goals

- Transparent recommendations (users understand why movies were suggested)
- Context-aware suggestions (date night ≠ family night)
- User control (explicit genre selection, keyword refinement)
- Rich presentation (posters, descriptions, metadata)
- Session adaptation (learn from swipes within session)

### Deliverables

1. Complete data pipeline (IMDb → TMDb → ML-ready dataset)
2. Trained ML models (CF, graph, content similarity)
3. REST API backend (Flask) with documented endpoints
4. React frontend with questionnaire and swipe interface
5. CLI tools for testing without frontend (movie_finder.py, interactive_movie_finder.py)
6. Documentation (README, API docs, architecture docs)
7. Project report (LaTeX, max 10 pages)

---

## Timeline

**Weeks 1-2: Data Pipeline**
- Download and clean IMDb datasets
- Implement TMDb API integration
- Feature engineering and data transformation
- Generate synthetic user interactions

**Weeks 3-4: ML Model Development**
- Implement collaborative filtering (ALS)
- Build co-occurrence graph
- Develop content similarity (TF-IDF)
- Train models and validate

**Weeks 5-6: Backend API**
- Implement Flask REST API
- Develop recommendation engine (smart_engine.py)
- Create keyword analyzer and filters
- Test API endpoints

**Weeks 7-8: Frontend Development**
- Design UI with Lovable (mockup tool)
- Implement React components
- Integrate with backend API
- Polish and test

**Weeks 9-10: Testing and Documentation**
- End-to-end testing
- CLI tools for easy demo
- Documentation writing
- Project report preparation

---

## Technical Challenges and Solutions

### Challenge 1: Cold-Start Problem
**Problem:** New users have no interaction history for collaborative filtering.
**Solution:** Generate 1000 synthetic users with realistic patterns to train CF model. Complement with content-based and graph-based recommendations.

### Challenge 2: Genre Matching
**Problem:** Initial version had only 40% genre match rate.
**Solution:** Explicit 25% weight for genre matching in scoring formula, ensuring 100% match rate.

### Challenge 3: TMDb API Rate Limits
**Problem:** 10 requests/second limit makes enrichment slow.
**Solution:** Batch processing, progress saving, 2-4 hour pipeline run time (acceptable for one-time setup).

### Challenge 4: Scalability
**Problem:** 45K × 45K similarity matrices are large.
**Solution:** Sparse matrix storage, pre-computation at training time, fast lookups at inference time.

### Challenge 5: Frontend-Backend Integration
**Problem:** Connecting React frontend to Python backend.
**Solution:** REST API with CORS, clear API client (api.ts), documented endpoints.

---

## Innovation and Contributions

1. **Explicit Genre Weighting**: Unlike typical systems, this explicitly weights user-selected genres at 25%, ensuring 100% match rate.

2. **Hybrid Approach**: Combines three distinct ML approaches (CF, graph, content) for robustness.

3. **Context Awareness**: Different scoring for different evening types (date night vs family night).

4. **Transparent System**: Users see why movies were recommended (genre match, quality, keywords).

5. **Full-Stack Implementation**: Complete system from data pipeline to deployed web interface.

---

## Tools and Technologies

**Development:**
- Claude Code: AI pair programming assistant
- Python 3.9+: Backend development
- React 18: Frontend development
- Git: Version control

**Libraries:**
- Backend: pandas, numpy, scipy, Flask, pyarrow
- Frontend: React, TypeScript, Vite, Tailwind CSS

**Data Sources:**
- IMDb: Public movie database
- TMDb: Metadata API (free tier)

---

## Conclusion

This project addresses a real problem (time wasted browsing streaming platforms) with a comprehensive technical solution combining data engineering, machine learning, and full-stack development. The hybrid approach ensures robust recommendations while respecting user intent, and the complete implementation demonstrates proficiency in advanced programming concepts.

Built with assistance from Claude Code, this project showcases modern development practices including AI-assisted coding, REST API design, and React frontend development.


