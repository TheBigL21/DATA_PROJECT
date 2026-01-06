import { Movie, MOVIES, QuestionnaireAnswers, SwipeFeedback, ERAS } from './movieData';

export class RecommendationEngine {
  private feedbackHistory: SwipeFeedback[] = [];
  private shownMovieIds: Set<string> = new Set();
  private answers: QuestionnaireAnswers | null = null;

  setAnswers(answers: QuestionnaireAnswers) {
    this.answers = answers;
    this.feedbackHistory = [];
    this.shownMovieIds = new Set();
  }

  addFeedback(feedback: SwipeFeedback) {
    this.feedbackHistory.push(feedback);
    this.shownMovieIds.add(feedback.movieId);
  }

  getNextRecommendation(): Movie | null {
    if (!this.answers) return null;

    const availableMovies = MOVIES.filter(m => !this.shownMovieIds.has(m.id));
    if (availableMovies.length === 0) return null;

    const scoredMovies = availableMovies.map(movie => ({
      movie,
      score: this.calculateScore(movie),
    }));

    scoredMovies.sort((a, b) => b.score - a.score);
    return scoredMovies[0]?.movie || null;
  }

  private calculateScore(movie: Movie): number {
    if (!this.answers) return 0;

    let score = 0;

    // Genre matching (30% weight)
    const genreMatches = movie.genres.filter(g => 
      this.answers!.genres.some(ag => ag.toLowerCase() === g.toLowerCase())
    ).length;
    score += (genreMatches / Math.max(this.answers.genres.length, 1)) * 30;

    // Keyword matching (25% weight)
    const keywordMatches = movie.keywords.filter(k =>
      this.answers!.keywords.some(ak => ak.toLowerCase() === k.toLowerCase())
    ).length;
    score += (keywordMatches / Math.max(this.answers.keywords.length, 1)) * 25;

    // Era matching (15% weight)
    const era = ERAS.find(e => e.id === this.answers!.era);
    if (era && movie.year >= era.range[0] && movie.year <= era.range[1]) {
      score += 15;
    }

    // Quality score based on rating (15% weight)
    score += (movie.rating / 10) * 15;

    // Learn from feedback (15% weight)
    const feedbackScore = this.calculateFeedbackInfluence(movie);
    score += feedbackScore * 15;

    return score;
  }

  private calculateFeedbackInfluence(movie: Movie): number {
    if (this.feedbackHistory.length === 0) return 0.5;

    let influence = 0;
    let count = 0;

    for (const feedback of this.feedbackHistory) {
      const feedbackMovie = MOVIES.find(m => m.id === feedback.movieId);
      if (!feedbackMovie) continue;

      // Check genre similarity
      const genreOverlap = movie.genres.filter(g => 
        feedbackMovie.genres.includes(g)
      ).length;

      // Check keyword similarity  
      const keywordOverlap = movie.keywords.filter(k =>
        feedbackMovie.keywords.includes(k)
      ).length;

      const similarity = (genreOverlap + keywordOverlap) / 
        (movie.genres.length + movie.keywords.length);

      // Weight by satisfaction
      influence += similarity * feedback.satisfaction;
      count++;
    }

    return count > 0 ? influence / count : 0.5;
  }

  getProgress(): { shown: number; total: number } {
    return {
      shown: this.shownMovieIds.size,
      total: MOVIES.length,
    };
  }

  hasMoreMovies(): boolean {
    return this.shownMovieIds.size < MOVIES.length;
  }
}

export const recommendationEngine = new RecommendationEngine();
