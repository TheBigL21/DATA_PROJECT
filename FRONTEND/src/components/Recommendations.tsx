import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation } from '@tanstack/react-query';
import { MovieCard } from './MovieCard';
import { Movie, QuestionnaireAnswers, mapApiMovieToMovie } from '@/lib/movieData';
import { apiClient, mapSwipeToBackendAction } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { RotateCcw, Film, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

interface RecommendationsProps {
  answers: QuestionnaireAnswers;
  onRestart: () => void;
}

export function Recommendations({ answers, onRestart }: RecommendationsProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [sessionHistory, setSessionHistory] = useState<Array<{ movie_id: number; action: 'yes' | 'no' | 'final' }>>([]);
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);
  const [sessionId] = useState(() => `session-${Date.now()}`);
  const [isFirstLoad, setIsFirstLoad] = useState(true);

  // Generate a user_id (in production, this would come from authentication)
  const userId = 1;

  // Fetch recommendations
  const { data: recommendations, isLoading, error } = useQuery({
    queryKey: ['recommendations', answers, sessionHistory],
    queryFn: () => apiClient.getRecommendations({
      user_id: userId,
      evening_type: answers.evening_type,
      genres: answers.genres,
      era: answers.era,
      keywords: answers.keywords,
      source_material: answers.source_material || 'any',
      session_history: sessionHistory,
      top_k: isFirstLoad ? 100 : 20, // 100 for initial load, 20 for updates
    }),
    enabled: !!answers.evening_type && answers.genres.length > 0 && !!answers.era,
  });

  // Mark first load as complete after first fetch
  useEffect(() => {
    if (recommendations && recommendations.length > 0 && isFirstLoad) {
      setIsFirstLoad(false);
    }
  }, [recommendations, isFirstLoad]);

  // Send feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: (feedback: { movie_id: number; action: 'yes' | 'no' | 'final' }) =>
      apiClient.sendFeedback({
        user_id: userId,
        movie_id: feedback.movie_id,
        action: feedback.action,
        session_id: sessionId,
      }),
    onError: (error) => {
      console.error('Failed to send feedback:', error);
      // Don't show error toast for feedback failures to avoid interrupting UX
    },
  });

  // Map API movies to frontend format
  const movies = recommendations?.map(mapApiMovieToMovie) || [];
  const currentMovie = movies[currentIndex] || null;

  // Reset index when new movies arrive
  useEffect(() => {
    if (movies.length > 0 && currentIndex >= movies.length) {
      setCurrentIndex(0);
    }
  }, [movies.length, currentIndex]);

  useEffect(() => {
    if (error) {
      toast.error('Failed to load recommendations. Please try again.');
    }
  }, [error]);

  const handleSwipe = (direction: 'left' | 'right' | 'up') => {
    if (!currentMovie) return;

    // Map swipe action to backend action format
    const backendAction = mapSwipeToBackendAction(direction);

    if (backendAction === 'final') {
      setSelectedMovie(currentMovie);
      feedbackMutation.mutate({
        movie_id: Number(currentMovie.id),
        action: 'final',
      });
      // Update session history
      const newHistory: Array<{ movie_id: number; action: 'yes' | 'no' | 'final' }> = [...sessionHistory, {
        movie_id: Number(currentMovie.id),
        action: 'final' as const,
      }];
      setSessionHistory(newHistory);
      return;
    }

    // Record feedback
    feedbackMutation.mutate({
      movie_id: Number(currentMovie.id),
      action: backendAction,
    });

    // Update session history with backend action format
    const newHistory: Array<{ movie_id: number; action: 'yes' | 'no' | 'final' }> = [...sessionHistory, {
      movie_id: Number(currentMovie.id),
      action: backendAction,
    }];
    setSessionHistory(newHistory);

    // Move to next movie
    if (currentIndex < movies.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
    // Note: When sessionHistory updates, React Query will automatically refetch
    // because sessionHistory is in the queryKey. The new movies will arrive
    // and the useEffect above will reset currentIndex if needed.
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
        <p className="mt-4 text-muted-foreground">Finding your perfect movies...</p>
      </div>
    );
  }

  if (error || !recommendations || recommendations.length === 0) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h2 className="question-text mb-4">No recommendations found</h2>
          <p className="text-muted-foreground mb-8">
            {error ? 'Failed to load recommendations. Please try again.' : 'We couldn\'t find any movies matching your preferences.'}
          </p>
          <Button variant="landing" size="lg" onClick={onRestart}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Start Over
          </Button>
        </motion.div>
      </div>
    );
  }

  if (selectedMovie) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="text-center max-w-md"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-blue-500 mb-8"
          >
            <Film className="w-10 h-10 text-white" />
          </motion.div>

          <h2 className="question-text mb-4">Perfect choice!</h2>
          <p className="text-muted-foreground mb-8 uppercase tracking-widest text-sm">
            You're watching
          </p>

          <div className="bg-card rounded-2xl shadow-elevated overflow-hidden mb-8">
            <img
              src={selectedMovie.poster}
              alt={selectedMovie.title}
              className="w-full aspect-[2/3] object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).src = '/placeholder.svg';
              }}
            />
            <div className="p-6">
              <h3 className="text-2xl font-semibold mb-2">{selectedMovie.title}</h3>
              <p className="text-muted-foreground text-sm">
                {selectedMovie.year} • {selectedMovie.runtime} min • ⭐ {selectedMovie.rating.toFixed(1)}
              </p>
            </div>
          </div>

          <Button variant="landingOutline" size="lg" onClick={onRestart}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Find Another Movie
          </Button>
        </motion.div>
      </div>
    );
  }

  if (!currentMovie) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h2 className="question-text mb-4">No more movies</h2>
          <p className="text-muted-foreground mb-8">
            You've seen all our recommendations!
          </p>
          <Button variant="landing" size="lg" onClick={onRestart}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Start Over
          </Button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 text-center"
      >
        <p className="text-muted-foreground uppercase tracking-widest text-sm">
          Movie {currentIndex + 1} of {movies.length}
        </p>
        {currentIndex === 0 && movies.length > 0 && (
          <p className="text-xs text-muted-foreground mt-2">
            Showing top recommendation from your preferences
          </p>
        )}
      </motion.div>

      <div className="w-full max-w-lg mx-auto px-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, scale: 0.8, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: -50 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          >
            <MovieCard
              movie={currentMovie}
              onSwipe={handleSwipe}
              showHints={currentIndex === 0}
            />
          </motion.div>
        </AnimatePresence>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="mt-12"
      >
        <Button variant="ghost" size="sm" onClick={onRestart}>
          <RotateCcw className="w-4 h-4 mr-2" />
          Start Over
        </Button>
      </motion.div>
    </div>
  );
}
