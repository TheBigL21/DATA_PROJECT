## Movie Finder — Hybrid Movie Recommendation System (Full-Stack)

Movie Finder is a context-aware movie recommendation system built to reduce the time people spend scrolling on streaming platforms without finding something that fits the moment.

- **A user-first interface**: you start by stating your intent (evening type, genres, era, optional themes)
- **A hybrid ML ranking engine**: collaborative filtering + co-occurrence graph + optional content similarity
- **Interactive feedback learning**: swipes are saved and used to adapt recommendations across sessions
- **A complete full-stack demo**: React frontend + Flask API backend + local data/model artifacts

This project was developed as an educational full-stack + data/ML system. The focus is not only on “predicting what’s popular”, but on building a transparent flow where the user stays in control and the system responds quickly with high-quality options.

---

## Project structure (what lives where)

This project follows the required university data science GitHub structure:

```
repo-root/
  README.md
  PROPOSAL.md
  environment.yml
  requirements.txt
  main.py              # Entry point (runs from repo root: python main.py)
  src/                 # Python source code package
    api_smart.py       # Flask API server
    run_pipeline.py    # Data pipeline orchestrator
    movie_finder.py    # CLI tools
    ...
    config/            # Configuration files
    data_processing/   # Data cleaning and transformation
    models/           # ML models
    recommendation/   # Recommendation engine
  data/
    raw/              # Raw data files (IMDb, etc.)
    processed/        # Processed data files (movies.parquet, etc.)
    models/           # Trained model artifacts (CF factors, graph, keyword DB, etc.)
    interactive_learning/  # Interactive learning models
    feedback.db       # SQLite database for user feedback
    *.parquet         # Additional processed data files
  frontend/           # React frontend (optional, separate from required structure)
```

### Backend (Python package in `src/`)
- **Role**: serves a Flask REST API and runs the recommendation engine.
- **What it loads at runtime**:
  - `data/processed/movies.parquet` (movie catalog + metadata)
  - `data/models/*` (trained model artifacts: CF factors, graph, optional keyword DB/content similarity)
  - `data/feedback.db` (SQLite database created/updated during usage)
  - `src/config/genre_allocation.json` (context → core/extended genre mapping)

Key entrypoints:
- `main.py` (repo root): thin entry point that imports from `src/` (API server by default; optional CLI and pipeline flags)
- `src/api_smart.py`: Flask app defining the API endpoints

### Frontend (`frontend/`)
- **Role**: React + TypeScript web UI that calls the backend API.
- **Dev server**: Vite on port **8080**
- **API configuration**: `VITE_API_URL` (defaults to `http://localhost:5000`)
- **Note**: Frontend is optional and separate from the required data science structure

Key file:
- `frontend/src/lib/api.ts`: API client used by the UI

---

## How the system works (end-to-end)

### Runtime flow (the “website path”)
1. **Frontend loads questionnaire options**
   - Calls `GET /api/questionnaire/options`
   - Displays evening types, eras, and available genres

2. **Frontend adapts genre choices to the selected context**
   - Calls `GET /api/questionnaire/genres?evening_type=...`
   - Shows “popular choices” and an optional extended list

3. **Frontend optionally requests theme suggestions**
   - Calls `POST /api/questionnaire/keywords` with selected genres
   - Calls `POST /api/questionnaire/source-material` to suggest an adaptation filter (optional)

4. **Frontend requests recommendations**
   - Calls `POST /api/recommend`
   - Backend generates a candidate pool (filters) and computes a hybrid score using:
     - explicit preferences (genres/era/themes)
     - collaborative filtering (ALS latent factors)
     - co-occurrence graph neighbors
     - optional content similarity (if available)
     - quality constraints (e.g., minimum vote counts)
     - feedback learning signals (if available)

5. **Frontend records swipe feedback**
   - Calls `POST /api/feedback`
   - Backend persists feedback to `data/feedback.db` so future sessions can adapt

### Backend endpoints used by the frontend
- `GET /health`
- `GET /api/questionnaire/options`
- `GET /api/questionnaire/genres?evening_type=...`
- `POST /api/questionnaire/keywords`
- `POST /api/questionnaire/source-material`
- `POST /api/recommend`
- `POST /api/feedback`
- `GET /api/movie/<movie_id>`

---

## Running the project locally (recommended demo path)

### Requirements
- **Python**: 3.9+ recommended
- **Node.js**: 18+ recommended
- **npm**: comes with Node

You will run **two processes** in two terminals:
- Backend on `http://localhost:5000`
- Frontend on `http://localhost:8080`

---

## 1) Backend setup (Quick Run: no pipeline)

### A) Ensure runtime artifacts exist
For grading/demo, the backend is designed to run from **precomputed artifacts** (movies + models). These files must exist in the repo root structure.

If you were given a "data package" zip:
- Extract it into the repo root so it creates:
  - `data/...` (models and processed data)
  - `src/config/genre_allocation.json`

Recommended extraction (safe + verifies):

```bash
python -m src.setup_from_package --package /path/to/movie_finder_data_package_*.zip --force
```

Or verify manually:

```bash
python -m src.verify_runtime_files
```

### B) Install Python dependencies

```bash
# From repo root
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

### C) Start the API server

```bash
# From repo root
python main.py
```

Health check:
- `http://localhost:5000/health`

---

## 2) Frontend setup

**Note**: The frontend is optional and separate from the required data science structure. It's included for demonstration purposes.

### A) Configure backend URL (optional)
If your backend is on the default port, you can skip this. Otherwise create `frontend/.env`:

```bash
echo "VITE_API_URL=http://localhost:5000" > frontend/.env
```

### B) Install and run

```bash
# From repo root
npm --prefix frontend install
npm --prefix frontend run dev
```

Open:
- `http://localhost:8080`

---

## Optional: Run the full data pipeline (slow, for completeness)

This rebuilds the dataset and retrains models. It can take hours, mostly due to TMDb rate limits and the size of the data.

High-level requirements:
- Raw IMDb `.tsv.gz` files (from `https://datasets.imdbws.com/`)
- A TMDb API key (set `TMDB_API_KEY` in your environment)

Example run (adjust paths to your machine):

```bash
# From repo root
python -m src.run_pipeline --imdb-dir data/raw/imdb_raw --output-dir ./data
```

For most demos and grading, Quick Run is the intended path.

---

## Troubleshooting

- **Frontend loads but shows API errors**
  - Confirm backend is running: `http://localhost:5000/health`
  - Confirm `VITE_API_URL` points to the correct backend URL/port

- **Backend crashes at startup with "missing files"**
  - Run: `python -m src.verify_runtime_files`
  - Fix by extracting the data package into repo root (or by running the full pipeline)

- **CORS/network errors in the browser**
  - Make sure you are calling the correct backend URL
  - Restart the frontend after changing `frontend/.env`

- **Port conflicts**
  - Backend: run `python main.py --port 8000` and set `VITE_API_URL=http://localhost:8000`
  - Frontend: edit `frontend/vite.config.ts` if you need a different port

---

## Notes for evaluation (what to look at)

- **Full-stack integration**: React UI calls Flask endpoints and renders live recommendations
- **Hybrid recommendation logic**: combines multiple signals instead of a single model
- **User intent is respected**: explicit genre/era choices drive filtering and scoring
- **Learning loop**: swipe feedback is persisted in SQLite and used to adapt future sessions

---

## License / context
Educational project developed for coursework (Advanced Programming 2025 — HEC Lausanne / UNIL).