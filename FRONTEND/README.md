# Movie Finder - Web Interface

A web port of the CLI `movie_finder.py` application. This React frontend consumes the Flask backend API to provide movie recommendations based on user preferences.

## Setup

### Backend Setup

1. Navigate to the `DATA_PROJECT` directory
2. Start the backend API server:
   ```bash
   cd DATA_PROJECT
   python3 api_smart.py
   ```
   The backend will run on `http://localhost:5000`

### Frontend Setup

1. Create a `.env` file in the `DATA_WEBSITE` directory:
   ```bash
   cd DATA_WEBSITE
   echo "VITE_API_URL=http://localhost:5000" > .env
   ```

2. Install dependencies:
   ```bash
   npm i
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will run on `http://localhost:8080`

