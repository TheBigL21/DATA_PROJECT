import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { QuestionnaireAnswers, EVENING_TYPE_MAP } from '@/lib/movieData';
import { apiClient, mapSwipeToBackendAction } from '@/lib/api';
import { ArrowLeft, ArrowRight, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

interface QuestionnaireProps {
  onComplete: (answers: QuestionnaireAnswers) => void;
}

type Step = 'evening_type' | 'genres' | 'era' | 'keywords';

const steps: Step[] = ['evening_type', 'genres', 'era', 'keywords'];

export function Questionnaire({ onComplete }: QuestionnaireProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState<Partial<QuestionnaireAnswers>>({
    evening_type: '',
    genres: [],
    keywords: [],
    era: '',
    source_material: undefined,
  });
  
  const [showExtendedGenres, setShowExtendedGenres] = useState(false);

  const step = steps[currentStep];

  // Fetch questionnaire options
  const { data: options, isLoading: isLoadingOptions, error: optionsError } = useQuery({
    queryKey: ['questionnaire-options'],
    queryFn: () => apiClient.getQuestionnaireOptions(),
  });

  // Fetch genre config when evening type is selected
  const { data: genreConfig, isLoading: isLoadingGenreConfig } = useQuery({
    queryKey: ['genre-config', answers.evening_type],
    queryFn: () => apiClient.getGenreConfig(answers.evening_type || ''),
    enabled: !!answers.evening_type && step === 'genres',
  });

  // Fetch keyword suggestions when genres are selected
  const { data: keywordSuggestions, isLoading: isLoadingKeywords } = useQuery({
    queryKey: ['keyword-suggestions', answers.genres],
    queryFn: () => apiClient.getKeywordSuggestions(answers.genres || []),
    enabled: (answers.genres?.length || 0) > 0 && step === 'keywords',
  });

  // Fetch source material when genres are selected
  const { data: sourceMaterialInfo, isLoading: isLoadingSourceMaterial } = useQuery({
    queryKey: ['source-material', answers.genres],
    queryFn: () => apiClient.getRelevantSourceMaterial(answers.genres || []),
    enabled: (answers.genres?.length || 0) > 0 && step === 'keywords',
  });

  useEffect(() => {
    if (optionsError) {
      toast.error('Failed to load questionnaire options. Please try again.');
    }
  }, [optionsError]);

  const handleEveningTypeSelect = (eveningTypeId: string) => {
    setAnswers(prev => ({ ...prev, evening_type: eveningTypeId }));
  };

  const handleGenreToggle = (genre: string) => {
    setAnswers(prev => ({
      ...prev,
      genres: prev.genres?.includes(genre)
        ? prev.genres.filter(g => g !== genre)
        : (prev.genres?.length || 0) < 2 ? [...(prev.genres || []), genre] : prev.genres,
    }));
  };

  const handleKeywordToggle = (keyword: string) => {
    setAnswers(prev => ({
      ...prev,
      keywords: prev.keywords?.includes(keyword)
        ? prev.keywords.filter(k => k !== keyword)
        : (prev.keywords?.length || 0) < 3 ? [...(prev.keywords || []), keyword] : prev.keywords,
    }));
  };

  const handleEraSelect = (eraId: string) => {
    setAnswers(prev => ({ ...prev, era: eraId }));
  };

  const handleSourceMaterialToggle = () => {
    if (sourceMaterialInfo) {
      if (answers.source_material === sourceMaterialInfo.source_material) {
        setAnswers(prev => ({ ...prev, source_material: undefined }));
      } else {
        setAnswers(prev => ({ ...prev, source_material: sourceMaterialInfo.source_material }));
      }
    }
  };

  const canProceed = () => {
    switch (step) {
      case 'evening_type': return !!answers.evening_type;
      case 'genres': return (answers.genres?.length || 0) >= 1;
      case 'era': return !!answers.era;
      case 'keywords': return true; // Keywords and source material are optional
      default: return false;
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      // Validate all required fields before completing
      if (answers.evening_type && answers.genres && answers.genres.length >= 1 && answers.era) {
        onComplete({
          evening_type: answers.evening_type,
          genres: answers.genres,
          keywords: answers.keywords || [],
          era: answers.era,
          source_material: answers.source_material,
        });
      }
    }
  };

  // Helper function to determine which fields to clear based on the destination step
  // When going back, we clear the destination step + all later steps, but keep earlier steps
  const getFieldsToClear = (destinationStep: Step): Partial<QuestionnaireAnswers> => {
    switch (destinationStep) {
      case 'evening_type':
        // Going back to evening_type: clear everything (there are no earlier steps)
        return {
          evening_type: '',
          genres: [],
          era: '',
          keywords: [],
          source_material: undefined,
        };
      case 'genres':
        // Going back to genres: keep evening_type, clear genres and all later steps
        return {
          genres: [],
          era: '',
          keywords: [],
          source_material: undefined,
        };
      case 'era':
        // Going back to era: keep evening_type and genres, clear era and all later steps
        return {
          era: '',
          keywords: [],
          source_material: undefined,
        };
      case 'keywords':
        // Going back to keywords: keep evening_type, genres, era, clear keywords and source_material
        return {
          keywords: [],
          source_material: undefined,
        };
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      const nextStepIndex = currentStep - 1;
      const destinationStep = steps[nextStepIndex];
      
      // Clear fields for the destination step and all later steps
      setAnswers(prev => ({
        ...prev,
        ...getFieldsToClear(destinationStep),
      }));
      
      // Reset extended genres view when going back to genres step
      if (destinationStep === 'genres') {
        setShowExtendedGenres(false);
      }
      
      setCurrentStep(nextStepIndex);
    }
  };

  const slideVariants = {
    enter: { opacity: 0, x: 50 },
    center: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -50 },
  };

  // Reset extended genres when evening type changes
  useEffect(() => {
    if (step === 'genres') {
      setShowExtendedGenres(false);
    }
  }, [answers.evening_type, step]);

  if (isLoadingOptions) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
        <p className="mt-4 text-muted-foreground">Loading...</p>
      </div>
    );
  }

  if (optionsError || !options) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
        <p className="text-destructive mb-4">Failed to load questionnaire options</p>
        <Button variant="outline" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative flex flex-col items-center justify-center px-6 py-12">
      {/* Back Button - Top Left */}
      {currentStep > 0 && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          className="absolute top-6 left-6 z-10"
        >
          <Button
            variant="landingGhost"
            size="icon"
            onClick={handleBack}
            aria-label="Back to previous question"
            className="h-10 w-10"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </motion.div>
      )}

      <div className="w-full max-w-3xl">
        {/* Progress */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-16"
        >
          <div className="flex justify-center gap-3">
            {steps.map((_, index) => (
              <div
                key={index}
                className={`h-1 w-16 rounded-full transition-colors duration-300 ${
                  index <= currentStep ? 'bg-primary' : 'bg-border'
                }`}
              />
            ))}
          </div>
        </motion.div>

        <AnimatePresence mode="wait">
          {step === 'evening_type' && (
            <motion.div
              key="evening_type"
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="text-center"
            >
              <h2 className="question-text mb-4">What's your plan for tonight?</h2>
              <p className="text-muted-foreground mb-12 uppercase tracking-widest text-sm">
                Choose one option
              </p>
              <div className="flex flex-wrap justify-center gap-3 mb-12">
                {options.evening_types.map(eveningType => (
                  <Button
                    key={eveningType.id}
                    variant={answers.evening_type === eveningType.id ? 'pillActive' : 'pill'}
                    size="lg"
                    onClick={() => handleEveningTypeSelect(eveningType.id)}
                    className="relative flex flex-col items-center p-6 h-auto"
                  >
                    <span className="text-lg font-semibold">{eveningType.label}</span>
                    <span className="text-xs text-muted-foreground mt-1">{eveningType.description}</span>
                  </Button>
                ))}
              </div>
            </motion.div>
          )}

          {step === 'genres' && (
            <motion.div
              key="genres"
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="text-center"
            >
              <h2 className="question-text mb-4">Pick your genres</h2>
              <p className="text-muted-foreground mb-12 uppercase tracking-widest text-sm">
                Select 1-2 genres
              </p>
              {isLoadingGenreConfig ? (
                <div className="flex justify-center mb-12">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              ) : (
                <>
                  {genreConfig && (
                    <>
                      <div className="mb-6">
                        <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wider">
                          Popular Choices
                        </h3>
                        <div className="flex flex-wrap justify-center gap-3 mb-6 max-w-xl mx-auto">
                          {genreConfig.core.map(genre => (
                            <Button
                              key={genre}
                              variant={answers.genres?.includes(genre) ? 'pillActive' : 'pill'}
                              size="lg"
                              onClick={() => handleGenreToggle(genre)}
                              className="relative capitalize"
                            >
                              {genre}
                            </Button>
                          ))}
                        </div>
                        {!showExtendedGenres && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setShowExtendedGenres(true)}
                            className="mt-4"
                          >
                            More genres...
                          </Button>
                        )}
                      </div>
                      {showExtendedGenres && genreConfig.extended && (
                        <div className="mt-8">
                          <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wider">
                            More Genres
                          </h3>
                          <div className="flex flex-wrap justify-center gap-3">
                            {genreConfig.extended.map(genre => (
                              <Button
                                key={genre}
                                variant={answers.genres?.includes(genre) ? 'pillActive' : 'pill'}
                                size="lg"
                                onClick={() => handleGenreToggle(genre)}
                                className="relative capitalize"
                              >
                                {genre}
                              </Button>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                  {!genreConfig && (
                    <div className="flex flex-wrap justify-center gap-3 mb-12">
                      {(options.genres && options.genres.length > 0 
                        ? options.genres 
                        : []
                      ).map(genre => (
                        <Button
                          key={genre}
                          variant={answers.genres?.includes(genre) ? 'pillActive' : 'pill'}
                          size="lg"
                          onClick={() => handleGenreToggle(genre)}
                          className="relative"
                        >
                          {genre}
                        </Button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </motion.div>
          )}

          {step === 'keywords' && (
            <motion.div
              key="keywords"
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="text-center"
            >
              <h2 className="question-text mb-4">What themes interest you?</h2>
              <p className="text-muted-foreground mb-12 uppercase tracking-widest text-sm">
                Select up to 3 themes (optional)
              </p>
              {(isLoadingKeywords || isLoadingSourceMaterial) ? (
                <div className="flex justify-center mb-12">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              ) : (
                <>
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wider">
                      Themes
                    </h3>
                    <div className="flex flex-wrap justify-center gap-3 mb-6 max-w-xl mx-auto">
                      {(keywordSuggestions ?? []).slice(0, 6).map(keyword => (
                        <Button
                          key={keyword}
                          variant={answers.keywords?.includes(keyword) ? 'pillActive' : 'pill'}
                          size="lg"
                          onClick={() => handleKeywordToggle(keyword)}
                          className="relative capitalize"
                        >
                          {keyword.replace(/_/g, ' ')}
                        </Button>
                      ))}
                    </div>
                  </div>
                  {sourceMaterialInfo && (
                    <div className="mt-8">
                      <div className="flex flex-wrap justify-center gap-3">
                        <Button
                          variant={answers.source_material === sourceMaterialInfo.source_material ? 'pillActive' : 'pill'}
                          size="lg"
                          onClick={handleSourceMaterialToggle}
                          className="relative"
                        >
                          {sourceMaterialInfo.label}
                        </Button>
                      </div>
                    </div>
                  )}
                  {keywordSuggestions && keywordSuggestions.length === 0 && !sourceMaterialInfo && (
                    <p className="text-muted-foreground mb-12">No suggestions available</p>
                  )}
                </>
              )}
            </motion.div>
          )}

          {step === 'era' && (
            <motion.div
              key="era"
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="text-center"
            >
              <h2 className="question-text mb-4">What time period are you in the mood for?</h2>
              <p className="text-muted-foreground mb-12 uppercase tracking-widest text-sm">
                Choose one era
              </p>
              <div className="flex flex-wrap justify-center gap-3 mb-12">
                {options.eras.map(era => (
                  <Button
                    key={era.id}
                    variant={answers.era === era.id ? 'pillActive' : 'pill'}
                    size="lg"
                    onClick={() => handleEraSelect(era.id)}
                    className="relative flex flex-col items-center p-6 h-auto"
                  >
                    <span className="text-lg font-semibold">{era.label}</span>
                    {era.description && (
                      <span className="text-xs text-muted-foreground mt-1">{era.description}</span>
                    )}
                  </Button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Continue Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className={`flex justify-center ${
            (step === 'genres' && showExtendedGenres) || step === 'keywords' ? 'mt-8' : ''
          }`}
        >
          <Button
            variant="landing"
            size="xl"
            onClick={handleNext}
            disabled={!canProceed()}
            className="min-w-[200px]"
          >
            {currentStep === steps.length - 1 ? 'Find Movies' : 'Continue'}
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </motion.div>
      </div>
    </div>
  );
}
