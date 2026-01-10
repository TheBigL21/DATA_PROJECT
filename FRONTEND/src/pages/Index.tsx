import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Landing } from '@/components/Landing';
import { Questionnaire } from '@/components/Questionnaire';
import { Recommendations } from '@/components/Recommendations';
import { QuestionnaireAnswers } from '@/lib/movieData';

type AppState = 'landing' | 'questionnaire' | 'recommendations';

const DEFAULT_USER_ID = 1000; // Default user when skipping

const Index = () => {
  const [appState, setAppState] = useState<AppState>('landing');
  const [answers, setAnswers] = useState<QuestionnaireAnswers | null>(null);
  const [userId, setUserId] = useState<number>(() => {
    // Try to load from localStorage, fallback to default
    const stored = localStorage.getItem('userId');
    return stored ? parseInt(stored, 10) : DEFAULT_USER_ID;
  });

  // Persist userId to localStorage when it changes
  useEffect(() => {
    if (userId) {
      localStorage.setItem('userId', userId.toString());
    }
  }, [userId]);

  const handleStart = (selectedUserId: number) => {
    setUserId(selectedUserId);
    setAppState('questionnaire');
  };

  const handleQuestionnaireComplete = (questionnaireAnswers: QuestionnaireAnswers) => {
    setAnswers(questionnaireAnswers);
    setAppState('recommendations');
  };

  const handleRestart = () => {
    setAnswers(null);
    setAppState('landing');
    // Keep userId on restart (don't clear localStorage)
  };

  return (
    <div className="min-h-screen bg-background">
      <AnimatePresence mode="wait">
        {appState === 'landing' && (
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Landing onStart={handleStart} />
          </motion.div>
        )}

        {appState === 'questionnaire' && (
          <motion.div
            key="questionnaire"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Questionnaire onComplete={handleQuestionnaireComplete} />
          </motion.div>
        )}

        {appState === 'recommendations' && answers && (
          <motion.div
            key="recommendations"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Recommendations answers={answers} userId={userId} onRestart={handleRestart} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Index;
