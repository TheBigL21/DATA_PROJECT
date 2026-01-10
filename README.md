<!-- Hero image (shown above the intro) -->
<p align="center">
  <img src="data/images/1rst%20interface.png" alt="Movie Finder - interface preview" width="900">
</p>

## Movie Finder — Hybrid Movie Recommendation System 

### Quick intro
If you’ve ever opened Netflix (or any streaming platform), scrolled for 15 minutes, and still felt undecided… that’s exactly the moment we built **Movie Finder** for.  
The idea is simple: **you tell the system what kind of evening you want**, and it gives you a short list of movie options that fit that context.

### The problem → our approach
- **Problem**: most recommendations are great at “popular right now”, but not always great at “what fits my mood / situation tonight”.
- **Our approach**: a short questionnaire + a **hybrid ranking engine** that combines multiple signals (instead of relying on a single model), plus a lightweight feedback loop.

---

## How it works (fast version)
1. You choose your **evening type** (date night, friends night, family night, chill solo).
2. You pick **1–2 genres**, an **era**, and optionally a few **themes**.
3. The backend ranks candidates using a mix of collaborative filtering + a co-occurrence graph (+ optional content similarity when available).
4. You browse recommendations and your feedback is saved locally (so the next session can improve).

---

## What’s included in this repo 
To make evaluation easy, the runtime artifacts are included directly in the repository under `data/` (around ~73MB).  
That means **you can run the project without downloading large datasets**.

At runtime, the backend loads:
- `data/processed/movies.parquet`
- `data/models/`
- `data/feedback.db` (created/updated automatically during usage)

---

## How to run the project (backend + frontend)

### Backend (Python)
From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

- Backend runs at `http://localhost:5000`
- Health check: `http://localhost:5000/health`

### Frontend (React)
In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

- Frontend runs at `http://localhost:8080`

---

## How to use it (what you’ll see)
<table>
  <tr>
    <td width="58%" valign="top">
      <p>
        Start by answering the questionnaire 
        (intent → genres → era → optional themes).<br><br>
        Then you can <b>swipe through recommendations</b> until you find a movie you actually want to watch.<br><br>
        Feedback is stored locally in <code>data/feedback.db</code>.
      </p>
    </td>
    <td width="42%" valign="top">
      <img src="data/images/swipe%20until%20you%20found%20your%20movie.png" alt="Swipe interface preview" width="420">
    </td>
  </tr>
</table>

---

## Project structure (quick map)
```
repo-root/
  main.py
  src/                     # backend source code (Flask API + recommendation engine)
  data/                    # included runtime artifacts (tracked in git)
    processed/
    models/
    interactive_learning/
    feedback.db
    images/
  frontend/                # React UI
```

---

## Troubleshooting (common issues)
- **Frontend shows API errors**
  - Check that the backend is running: `http://localhost:5000/health`
  - Restart the frontend if you changed ports

- **Backend complains about missing files**
  - Confirm these exist:
    - `data/processed/movies.parquet`
    - `data/models/`

- **Port conflict**
  - Start backend on another port:
    ```bash
    python main.py --port 8000
    ```

---

## Where the main logic is
- `src/api_smart.py`: Flask endpoints used by the frontend
- `src/recommendation/smart_engine.py`: hybrid scoring and ranking
- `src/movie_finder.py` / `src/interactive_movie_finder.py`: CLI modes (optional)

---

## Optional: rebuild data/models (not required for the demo)
There is a full pipeline that can rebuild the dataset and retrain models, but it’s slow (hours) and requires downloading IMDb files + a TMDb API key.  
For grading/demo, the intended path is using the included `data/` artifacts.