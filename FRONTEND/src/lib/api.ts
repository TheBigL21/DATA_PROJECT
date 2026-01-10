// Use empty string for relative URLs - Vite proxy handles routing
const API_BASE_URL = '';

export interface ApiMovie {
  movie_id: number;
  title: string;
  year: number;
  runtime: number;
  genres: string[];
  avg_rating: number;
  tmdb_rating?: number;
  combined_rating: number;
  num_votes: number;
  director: string;
  actors: string[];
  description?: string;
  poster_url?: string;
  keywords?: string[];
}

export interface EveningType {
  id: string;
  label: string;
  description: string;
}

export interface Era {
  id: string;
  label: string;
  description: string;
  years: string;
}

export interface QuestionnaireOptions {
  evening_types: EveningType[];
  genres: string[];
  eras: Era[];
}

export interface GenreConfig {
  core: string[];
  extended: string[];
}

export interface SourceMaterialInfo {
  source_material: string;
  label: string;
  description: string;
}

export interface RecommendRequest {
  user_id: number;
  evening_type: string;
  genres: string[];
  era: string;
  keywords?: string[];
  source_material?: string;
  session_history?: Array<{ movie_id: number; action: 'yes' | 'no' | 'final' }>;
  session_id?: string;
  top_k?: number;
}

export interface FeedbackRequest {
  user_id: number;
  movie_id: number;
  action: 'yes' | 'no' | 'final';
  session_id?: string;
  timestamp?: string;
  genres?: string[];
  era?: string;
  themes?: string[];
  position_in_session?: number;
  previous_movie_id?: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  }

  async getQuestionnaireOptions(): Promise<QuestionnaireOptions> {
    const response = await fetch(`${this.baseUrl}/api/questionnaire/options`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch questionnaire options' }));
      throw new Error(error.error || 'Failed to fetch questionnaire options');
    }
    return response.json();
  }

  async getGenreConfig(eveningType: string): Promise<GenreConfig> {
    const response = await fetch(`${this.baseUrl}/api/questionnaire/genres?evening_type=${eveningType}`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch genre config' }));
      throw new Error(error.error || 'Failed to fetch genre config');
    }
    return response.json();
  }

  async getRelevantSourceMaterial(genres: string[]): Promise<SourceMaterialInfo> {
    const response = await fetch(`${this.baseUrl}/api/questionnaire/source-material`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genres }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch source material' }));
      throw new Error(error.error || 'Failed to fetch source material');
    }
    return response.json();
  }

  async getKeywordSuggestions(genres: string[]): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/questionnaire/keywords`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genres }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch keyword suggestions' }));
      throw new Error(error.error || 'Failed to fetch keyword suggestions');
    }
    const data = await response.json();
    return data.keywords || [];
  }

  async getRecommendations(request: RecommendRequest): Promise<ApiMovie[]> {
    const response = await fetch(`${this.baseUrl}/api/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch recommendations' }));
      throw new Error(error.error || 'Failed to fetch recommendations');
    }
    const data = await response.json();
    return data.recommendations || [];
  }

  async getMovie(movieId: number): Promise<ApiMovie> {
    const response = await fetch(`${this.baseUrl}/api/movie/${movieId}`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch movie' }));
      throw new Error(error.error || 'Failed to fetch movie');
    }
    return response.json();
  }

  async sendFeedback(feedback: FeedbackRequest): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedback),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to send feedback' }));
      throw new Error(error.error || 'Failed to send feedback');
    }
  }

  async signup(displayCode: string): Promise<{ user_id: number; display_code: string }> {
    const response = await fetch(`${this.baseUrl}/api/users/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_code: displayCode }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to sign up' }));
      throw new Error(error.error || 'Failed to sign up');
    }
    return response.json();
  }

  async login(displayCode: string): Promise<{ user_id: number; display_code: string }> {
    const response = await fetch(`${this.baseUrl}/api/users/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_code: displayCode }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to login' }));
      throw new Error(error.error || 'Failed to login');
    }
    return response.json();
  }
}

export const apiClient = new ApiClient(API_BASE_URL);

/**
 * Map frontend swipe actions to backend action format
 */
export function mapSwipeToBackendAction(direction: 'left' | 'right' | 'up'): 'yes' | 'no' | 'final' {
  const map: Record<'left' | 'right' | 'up', 'yes' | 'no' | 'final'> = {
    left: 'no',
    right: 'yes',
    up: 'final',
  };
  return map[direction];
}

