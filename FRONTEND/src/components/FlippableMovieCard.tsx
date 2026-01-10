import { useState } from 'react';
import { motion } from 'framer-motion';
import { Movie } from '@/lib/movieData';
import { Star, Clock, User, BookOpen, Film } from 'lucide-react';

interface FlippableMovieCardProps {
  movie: Movie;
}

export function FlippableMovieCard({ movie }: FlippableMovieCardProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div 
      className="relative w-full max-w-sm mx-auto h-[600px]"
      style={{ perspective: '1000px' }}
    >
      <motion.div
        className="relative w-full h-full"
        animate={{ rotateY: isFlipped ? 180 : 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        style={{ transformStyle: 'preserve-3d' }}
      >
        {/* Front Side - Poster and Basic Info */}
        <motion.div
          className="absolute inset-0 w-full h-full rounded-2xl overflow-hidden shadow-elevated bg-card cursor-pointer flex flex-col"
          style={{ backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden' }}
          onClick={() => setIsFlipped(true)}
        >
          <div className="flex-1 relative min-h-0">
            <img
              src={movie.poster}
              alt={movie.title}
              className="w-full h-full object-cover"
              draggable={false}
              onError={(e) => {
                (e.target as HTMLImageElement).src = '/placeholder.svg';
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-card via-transparent to-transparent" />
          </div>

          <div className="p-6 bg-card">
            <div className="flex items-center gap-2 mb-2">
              <span className="bg-accent text-accent-foreground px-2 py-1 rounded text-xs font-medium uppercase tracking-wider">
                {movie.year}
              </span>
              <div className="flex items-center gap-1 text-accent">
                <Star className="w-4 h-4 fill-current" />
                <span className="text-sm font-medium">{movie.rating.toFixed(2)}</span>
              </div>
            </div>

            <h3 className="text-2xl font-semibold mb-2 text-card-foreground">{movie.title}</h3>

            <div className="flex items-center gap-4 text-muted-foreground text-sm mb-2">
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{movie.runtime} min</span>
              </div>
              <div className="flex items-center gap-1">
                <User className="w-4 h-4" />
                <span className="truncate max-w-[150px]">{movie.director}</span>
              </div>
            </div>

            <p className="text-xs text-muted-foreground mt-3 text-center opacity-80">
              Tap to see full details
            </p>
          </div>
        </motion.div>

        {/* Back Side - Full Details */}
        <motion.div
          className="absolute inset-0 w-full h-full rounded-2xl overflow-hidden shadow-elevated bg-card cursor-pointer"
          style={{ 
            backfaceVisibility: 'hidden', 
            WebkitBackfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)'
          }}
          onClick={() => setIsFlipped(false)}
        >
          <div className="h-full overflow-y-auto p-6">
            {/* Close/Flip indicator */}
            <div className="flex justify-end mb-4">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsFlipped(false);
                }}
                className="text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Flip back"
              >
                <Film className="w-5 h-5" />
              </button>
            </div>

            {/* Title */}
            <h3 className="text-2xl font-semibold mb-4 text-card-foreground">{movie.title}</h3>

            {/* Basic Info Row */}
            <div className="flex items-center gap-4 text-muted-foreground text-sm mb-4 pb-4 border-b border-border">
              <div className="flex items-center gap-1">
                <span className="bg-accent text-accent-foreground px-2 py-1 rounded text-xs font-medium uppercase tracking-wider">
                  {movie.year}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{movie.runtime} min</span>
              </div>
              <div className="flex items-center gap-1">
                <Star className="w-4 h-4 fill-current text-accent" />
                <span className="font-medium">{movie.rating.toFixed(2)}</span>
              </div>
            </div>

            {/* Director */}
            <div className="mb-4">
              <div className="flex items-center gap-2 text-sm mb-1">
                <User className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground font-medium">Director</span>
              </div>
              <p className="text-card-foreground">{movie.director}</p>
            </div>

            {/* Genres */}
            <div className="mb-4">
              <div className="flex items-center gap-2 text-sm mb-2">
                <Film className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground font-medium">Genres</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {movie.genres.map(genre => (
                  <span
                    key={genre}
                    className="text-xs uppercase tracking-wider text-muted-foreground border border-border px-2 py-1 rounded bg-secondary/50"
                  >
                    {genre}
                  </span>
                ))}
              </div>
            </div>

            {/* Keywords (if available) */}
            {movie.keywords && movie.keywords.length > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 text-sm mb-2">
                  <BookOpen className="w-4 h-4 text-muted-foreground" />
                  <span className="text-muted-foreground font-medium">Themes</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {movie.keywords.slice(0, 8).map((keyword, idx) => (
                    <span
                      key={idx}
                      className="text-xs capitalize text-muted-foreground border border-border/50 px-2 py-1 rounded bg-background/50"
                    >
                      {keyword.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Full Description */}
            <div className="mb-4">
              <div className="flex items-center gap-2 text-sm mb-2">
                <BookOpen className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground font-medium">Description</span>
              </div>
              <p className="text-muted-foreground text-sm leading-relaxed">
                {movie.description || 'No description available.'}
              </p>
            </div>

            {/* Flip back hint */}
            <p className="text-xs text-muted-foreground mt-4 text-center opacity-80 pt-4 border-t border-border">
              Tap anywhere to flip back
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
