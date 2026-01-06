# Movie Finder: Hybrid Movie Recommendation System

**Advanced Programming 2025 - HEC Lausanne / UNIL**

<p align="center">
  <img src="images/hero.png" alt="Movie Finder landing page" width="900">
</p>

## Introduction

Movie Finder is a hybrid recommendation system that helps you quickly choose a movie that matches your current mood (date night, friends night, family night, or a quiet evening alone). I built it because I personally enjoy watching movies, but I often spent too much time browsing without finding something that fits the moment.

The system respects explicit user intent (genres, era, optional themes), enforces basic quality constraints, and learns from swipe feedback during and across sessions (stored in a local SQLite database).

**Key features**
- Hybrid ranking: collaborative filtering (ALS), co-occurrence graph, optional TF--IDF content similarity
- Persistent feedback learning (SQLite) with context-specific adaptation
- IMDb dataset enriched with TMDb metadata (keywords, posters, descriptions, extra ratings)
- React frontend with swipe-based interaction

---

## Run without pipeline (recommended)

This is the easiest way to run the project for grading/demo: you use **precomputed data + trained models**, so you do **not** need to download raw IMDb files or fetch TMDb (which can take hours).

### 1) Get and extract the data package (zip)

You should have the following files inside `DATA_PROJECT/`:

**Required runtime files**
- `output/processed/movies.parquet`
- `output/models/user_factors.npy`
- `output/models/movie_factors.npy`
- `output/models/model_metadata.json`
- `output/models/adjacency_matrix.npz`
- `output/models/graph_metadata.json`
- `config/genre_allocation.json`

**Optional (system still runs without them)**
- `output/models/keyword_database.pkl` (keyword suggestions)
- `output/models/content_similarity.pkl` (content similarity signal)
- `output/feedback.db` (can be empty; if missing, it will be created on first run)

**Where to get the package**
I provide a `.zip` (for example as a GitHub Release asset / download link) containing only the runtime artifacts.

**Extract it into `DATA_PROJECT/`** so it creates `DATA_PROJECT/output/...` and includes `DATA_PROJECT/config/genre_allocation.json`.

Example (macOS/Linux):

```bash
cd DATA_PROJECT
unzip /path/to/movie_finder_data_package_*.zip -d .
```

Alternative (recommended, handles overwrite + verifies):

```bash
cd DATA_PROJECT
python setup_from_package.py --package /path/to/movie_finder_data_package_*.zip --force
```

Then verify:

```bash
cd DATA_PROJECT
python verify_runtime_files.py
```

### 2) Start the backend API

```bash
cd DATA_PROJECT
pip install -r requirements.txt
python main.py
```

Backend health check:
- `http://localhost:5000/health`

### 3) Start the frontend

```bash
cd DATA_WEBSITE
npm install
npm run dev
```

Frontend URL:
- `http://localhost:5173`

If your backend runs on a different host/port, set `VITE_API_URL` (environment variable or `DATA_WEBSITE/.env`), for example:

```bash
VITE_API_URL=http://localhost:5000
```

**Windows notes**
- Use PowerShell `Expand-Archive` or a GUI unzip tool to extract the zip into `DATA_PROJECT\`.
- If you change backend port, update `VITE_API_URL` accordingly.

---

## Run full pipeline (optional)

This option rebuilds the dataset and trains models from scratch. It requires raw IMDb files and a TMDb API key. It can take hours mainly due to TMDb rate limits.

### Prerequisites (pipeline)
- TMDb API key (recommended via environment variable `TMDB_API_KEY`)
- Raw IMDb `.tsv.gz` files (download from `https://datasets.imdbws.com/`)

### Run

```bash
cd DATA_PROJECT
pip install -r requirements.txt
python main.py --pipeline
```

---

## System Architecture (high level)

At runtime, the engine:
1. Builds a **candidate pool** using hard filters (genre match, rating threshold, vote threshold, era range, not already shown).
2. Computes a **hybrid score** for each candidate combining:
   - explicit preference scores,
   - collaborative filtering (ALS),
   - co-occurrence graph neighbors,
   - optional TF--IDF content similarity,
   - feedback/compatibility adjustments from SQLite.

The system is designed for interactive use (fast enough to swipe through recommendations). Actual latency depends on machine and dataset/model sizes.

---

## API Endpoints (backend)

- `GET /health`
- `GET /api/questionnaire/options`
- `GET /api/questionnaire/genres?evening_type=...`
- `POST /api/questionnaire/keywords`
- `POST /api/questionnaire/source-material`
- `POST /api/recommend`
- `POST /api/feedback`
- `GET /api/movie/<id>`

---

## Project structure (key files)

```
DATA_PROJECT/
  main.py                         # Entrypoint (API / CLI / pipeline)
  api_smart.py                     # Flask API server
  src/recommendation/smart_engine.py# SmartRecommendationEngine (main ranking logic)
  src/models/                       # CF / graph / optional content similarity code
  output/                           # Precomputed artifacts (movies, models, feedback DB)

DATA_WEBSITE/
  src/lib/api.ts                    # API client (uses VITE_API_URL or localhost:5000)
```

---

## Troubleshooting

- **Frontend cannot reach backend**: check `http://localhost:5000/health` and set `VITE_API_URL` if needed.
- **Port already in use**: run `python main.py --port 8000` (and update `VITE_API_URL`).
- **Models not found**: you are missing the data package; either extract it into `DATA_PROJECT/output/` or run the pipeline.
- **Quick Run check**: run `python verify_runtime_files.py` inside `DATA_PROJECT/` to see exactly what is missing.
- **CORS errors**: ensure backend is running and you are calling the correct URL/port.
- **Pipeline fails on TMDb**: confirm `TMDB_API_KEY` is set and that you are respecting rate limits.

---

## Limitations (honest notes)

- **Cold-start**: the collaborative filtering model is trained on synthetic interactions, so new users can see similar initial recommendations until real feedback accumulates.
- **Popularity bias**: vote thresholds improve quality but can reduce niche discovery.
- **External dependency**: the full pipeline depends on TMDb and is slowed by rate limits.

---

## License

Educational project for Advanced Programming 2025. Not for commercial use.
