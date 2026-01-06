import { motion, PanInfo, useMotionValue, useTransform, AnimatePresence } from 'framer-motion';
import { Movie } from '@/lib/movieData';
import { Star, Clock, User, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-react';
import { useState, useEffect } from 'react';

interface MovieCardProps {
  movie: Movie;
  onSwipe: (direction: 'left' | 'right' | 'up') => void;
  showHints: boolean;
}

export function MovieCard({ movie, onSwipe, showHints }: MovieCardProps) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const [showInitialHints, setShowInitialHints] = useState(showHints);

  const rotateZ = useTransform(x, [-200, 200], [-15, 15]);
  const opacity = useTransform(
    x,
    [-200, -100, 0, 100, 200],
    [0.5, 0.8, 1, 0.8, 0.5]
  );

  // Swipe indicators with grey styling - fade smoothly
  const leftIndicatorOpacity = useTransform(x, [-200, -100, -50, 0], [1, 0.8, 0.3, 0]);
  const rightIndicatorOpacity = useTransform(x, [0, 50, 100, 200], [0, 0.3, 0.8, 1]);
  const upIndicatorOpacity = useTransform(y, [-200, -100, -50, 0], [1, 0.8, 0.3, 0]);
  
  // Scale transforms for cool effect
  const leftScale = useTransform(x, [-200, -100, 0], [1.15, 1.05, 1]);
  const rightScale = useTransform(x, [0, 100, 200], [1, 1.05, 1.15]);
  const upScale = useTransform(y, [-200, -100, 0], [1.15, 1.05, 1]);

  // Hide initial hints after 3 seconds
  useEffect(() => {
    if (showHints) {
      const timer = setTimeout(() => {
        setShowInitialHints(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [showHints]);

  const handleDragEnd = (_: any, info: PanInfo) => {
    const threshold = 100;
    const velocityThreshold = 500;

    if (info.offset.y < -threshold || info.velocity.y < -velocityThreshold) {
      onSwipe('up');
    } else if (info.offset.x > threshold || info.velocity.x > velocityThreshold) {
      onSwipe('right');
    } else if (info.offset.x < -threshold || info.velocity.x < -velocityThreshold) {
      onSwipe('left');
    }
  };

  return (
    <div className="relative w-full max-w-sm mx-auto">
      {/* Animated Swipe Hints - Only for first movie, fade out after 3 seconds */}
      <AnimatePresence>
        {showInitialHints && (
          <div className="absolute inset-0 pointer-events-none z-20">
          {/* Left - Nope (lower position) */}
          <motion.div
            key="hint-left"
            initial={{ opacity: 0, x: -20, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: -20, scale: 0.8 }}
            transition={{ 
              delay: 0.3, 
              duration: 0.6,
              ease: [0.22, 1, 0.36, 1],
              exit: { duration: 0.5, ease: [0.22, 1, 0.36, 1] }
            }}
            className="absolute -left-20 top-[65%] -translate-y-1/2 text-center"
          >
            <motion.div 
              className="bg-secondary border border-border rounded-full px-6 py-4 shadow-elevated"
              animate={{ 
                scale: [1, 1.05, 1],
              }}
              transition={{ 
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <ArrowLeft className="w-8 h-8 mx-auto mb-2 text-secondary-foreground" />
              <p className="text-sm font-medium uppercase tracking-wider text-secondary-foreground">Nope</p>
            </motion.div>
          </motion.div>

          {/* Right - Yes (higher position) */}
          <motion.div
            key="hint-right"
            initial={{ opacity: 0, x: 20, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.8 }}
            transition={{ 
              delay: 0.5, 
              duration: 0.6,
              ease: [0.22, 1, 0.36, 1],
              exit: { duration: 0.5, ease: [0.22, 1, 0.36, 1] }
            }}
            className="absolute -right-20 top-[35%] -translate-y-1/2 text-center"
          >
            <motion.div 
              className="bg-secondary border border-border rounded-full px-6 py-4 shadow-elevated"
              animate={{ 
                scale: [1, 1.05, 1],
              }}
              transition={{ 
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.2
              }}
            >
              <ArrowRight className="w-8 h-8 mx-auto mb-2 text-secondary-foreground" />
              <p className="text-sm font-medium uppercase tracking-wider text-secondary-foreground">Yes</p>
            </motion.div>
          </motion.div>

          {/* Up - That's the one! (more to the right) */}
          <motion.div
            key="hint-up"
            initial={{ opacity: 0, y: -20, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.8 }}
            transition={{ 
              delay: 0.7, 
              duration: 0.6,
              ease: [0.22, 1, 0.36, 1],
              exit: { duration: 0.5, ease: [0.22, 1, 0.36, 1] }
            }}
            className="absolute left-[60%] -translate-x-1/2 -top-20 text-center"
          >
            <motion.div 
              className="bg-secondary border border-border rounded-full px-6 py-4 shadow-elevated"
              animate={{ 
                scale: [1, 1.05, 1],
              }}
              transition={{ 
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.4
              }}
            >
              <ArrowUp className="w-8 h-8 mx-auto mb-2 text-secondary-foreground" />
              <p className="text-sm font-medium uppercase tracking-wider text-secondary-foreground">That's the one!</p>
            </motion.div>
          </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Swipe Indicators - Grey styling, text only, smaller, fades away */}
      <motion.div
        style={{ opacity: leftIndicatorOpacity }}
        className="absolute inset-0 bg-secondary/30 backdrop-blur-sm rounded-2xl z-10 pointer-events-none flex items-center justify-center"
      >
        <motion.div
          style={{ scale: leftScale }}
          className="text-center"
        >
          <span className="text-secondary-foreground text-3xl font-bold uppercase tracking-widest block">
            Nope
          </span>
        </motion.div>
      </motion.div>

      <motion.div
        style={{ opacity: rightIndicatorOpacity }}
        className="absolute inset-0 bg-secondary/30 backdrop-blur-sm rounded-2xl z-10 pointer-events-none flex items-center justify-center"
      >
        <motion.div
          style={{ scale: rightScale }}
          className="text-center"
        >
          <span className="text-secondary-foreground text-3xl font-bold uppercase tracking-widest block">
            Yes
          </span>
        </motion.div>
      </motion.div>

      <motion.div
        style={{ opacity: upIndicatorOpacity }}
        className="absolute inset-0 bg-secondary/30 backdrop-blur-sm rounded-2xl z-10 pointer-events-none flex items-center justify-center"
      >
        <motion.div
          style={{ scale: upScale }}
          className="text-center"
        >
          <span className="text-secondary-foreground text-3xl font-bold uppercase tracking-widest block">
            That's the one!
          </span>
        </motion.div>
      </motion.div>

      {/* Card */}
      <motion.div
        drag
        dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
        dragElastic={0.7}
        onDragEnd={handleDragEnd}
        style={{ x, y, rotateZ, opacity }}
        whileTap={{ cursor: 'grabbing' }}
        className="relative bg-card rounded-2xl shadow-elevated overflow-hidden cursor-grab"
      >
        {/* Poster */}
        <div className="aspect-[2/3] relative">
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

        {/* Info */}
        <div className="p-6 -mt-20 relative z-10">
          <div className="flex items-center gap-2 mb-2">
            <span className="bg-accent text-accent-foreground px-2 py-1 rounded text-xs font-medium uppercase tracking-wider">
              {movie.year}
            </span>
            <div className="flex items-center gap-1 text-accent">
              <Star className="w-4 h-4 fill-current" />
              <span className="text-sm font-medium">{movie.rating}</span>
            </div>
          </div>

          <h2 className="text-2xl font-semibold mb-2 text-card-foreground">{movie.title}</h2>

          <div className="flex items-center gap-4 text-muted-foreground text-sm mb-4">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{movie.runtime} min</span>
            </div>
            <div className="flex items-center gap-1">
              <User className="w-4 h-4" />
              <span>{movie.director}</span>
            </div>
          </div>

          <p className="text-muted-foreground text-sm leading-relaxed line-clamp-3">
            {movie.description}
          </p>

          <div className="flex flex-wrap gap-2 mt-4">
            {movie.genres.slice(0, 3).map(genre => (
              <span
                key={genre}
                className="text-xs uppercase tracking-wider text-muted-foreground border border-border px-2 py-1 rounded"
              >
                {genre}
              </span>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
