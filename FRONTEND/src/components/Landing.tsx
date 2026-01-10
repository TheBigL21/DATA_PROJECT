import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Film } from 'lucide-react';
import { apiClient } from '@/lib/api';

type AuthMode = 'initial' | 'login' | 'signup' | 'confirmed';

interface LandingProps {
  onStart: (userId: number) => void;
}

export function Landing({ onStart }: LandingProps) {
  const [authMode, setAuthMode] = useState<AuthMode>('initial');
  const [userIdInput, setUserIdInput] = useState('');
  const [confirmedUserId, setConfirmedUserId] = useState<string | null>(null);
  const [wasSignup, setWasSignup] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when auth mode changes
  useEffect(() => {
    if (authMode !== 'initial' && inputRef.current) {
      inputRef.current.focus();
    }
  }, [authMode]);

  // Clear error after 3 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleSkip = () => {
    // Default user 1000
    onStart(1000);
  };

  const handleUserIdInput = (digit: string) => {
    // Block if first digit is 0
    if (userIdInput.length === 0 && digit === '0') {
      setError('User ID must start with 1-9');
      return;
    }

    // Only allow digits and max 4 digits
    if (digit.match(/^\d$/) && userIdInput.length < 4) {
      setUserIdInput(prev => prev + digit);
      setError(null);
    }
  };

  const handleBackspace = () => {
    setUserIdInput(prev => prev.slice(0, -1));
    setError(null);
  };

  const handleLogin = async () => {
    if (userIdInput.length !== 4) {
      setError('Please enter a 4-digit User ID');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await apiClient.login(userIdInput);
      localStorage.setItem('userId', result.user_id.toString());
      localStorage.setItem('displayCode', result.display_code);
      setConfirmedUserId(result.display_code);
      setWasSignup(false);
      setAuthMode('confirmed');
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || 'Login failed');
      setIsLoading(false);
    }
  };

  const handleSignup = async () => {
    if (userIdInput.length !== 4) {
      setError('Please enter a 4-digit User ID');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await apiClient.signup(userIdInput);
      localStorage.setItem('userId', result.user_id.toString());
      localStorage.setItem('displayCode', result.display_code);
      setConfirmedUserId(result.display_code);
      setWasSignup(true);
      setAuthMode('confirmed');
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || 'Signup failed');
      setIsLoading(false);
    }
  };

  const handleContinue = () => {
    if (confirmedUserId) {
      const userId = parseInt(confirmedUserId, 10);
      onStart(userId);
    }
  };

  const switchToSignup = () => {
    setAuthMode('signup');
    setUserIdInput('');
    setError(null);
  };

  const switchToLogin = () => {
    setAuthMode('login');
    setUserIdInput('');
    setError(null);
  };

  const backToInitial = () => {
    setAuthMode('initial');
    setUserIdInput('');
    setError(null);
  };

  // Number pad for input
  const numberPad = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '←', '✓'];

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        className="text-center max-w-2xl w-full"
      >
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-8"
        >
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary mb-6">
            <Film className="w-10 h-10 text-primary-foreground" />
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-5xl md:text-7xl font-light tracking-tight mb-4"
        >
          MOVIE FINDER
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="text-muted-foreground text-lg md:text-xl mb-12 font-light"
        >
          Find Your Perfect Movie Tonight
        </motion.p>

        {/* Auth UI Container */}
        <div className="relative min-h-[400px] flex items-center justify-center">
          <AnimatePresence mode="wait">
            {authMode === 'initial' && (
              <motion.div
                key="initial-buttons"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="flex flex-col sm:flex-row gap-4 justify-center items-center"
              >
                <Button variant="landing" size="lg" onClick={() => setAuthMode('login')}>
                  Login
                </Button>
                <Button variant="landingOutline" size="lg" onClick={() => setAuthMode('signup')}>
                  Sign Up
                </Button>
                <Button variant="landingGhost" size="lg" onClick={handleSkip}>
                  Skip
                </Button>
              </motion.div>
            )}

            {/* Login/Signup input card */}
            {(authMode === 'login' || authMode === 'signup') && (
              <motion.div
                key="auth-card"
                initial={{ opacity: 0, scale: 0.9, x: 50 }}
                animate={{ opacity: 1, scale: 1, x: 0 }}
                exit={{ opacity: 0, scale: 0.9, x: 50 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="bg-card rounded-2xl shadow-elevated p-8 max-w-md mx-auto w-full"
              >
              <h2 className="text-2xl font-semibold mb-2 text-card-foreground">
                {authMode === 'login' ? 'Login' : 'Sign Up'}
              </h2>
              <p className="text-muted-foreground mb-6 text-sm">
                {authMode === 'login' 
                  ? 'Enter your 4-digit User ID' 
                  : 'Choose a 4-digit User ID (1000-9999)'}
              </p>

              {/* User ID display (circle with underscores) */}
              <div className="flex justify-center gap-3 mb-8">
                {[0, 1, 2, 3].map((index) => (
                  <motion.div
                    key={index}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: index * 0.1, duration: 0.3 }}
                    className={`w-16 h-16 rounded-full border-2 flex items-center justify-center text-3xl font-semibold transition-colors ${
                      userIdInput[index] 
                        ? 'border-primary bg-primary/10 text-primary' 
                        : 'border-border bg-secondary/50 text-muted-foreground'
                    }`}
                  >
                    {userIdInput[index] || '_'}
                  </motion.div>
                ))}
              </div>

              {/* Hidden input for keyboard events (for actual keyboard typing) */}
              <input
                ref={inputRef}
                type="text"
                inputMode="numeric"
                pattern="[1-9][0-9]{3}"
                maxLength={4}
                value={userIdInput}
                onChange={(e) => {
                  let value = e.target.value.replace(/\D/g, '');
                  
                  // Block if first digit is 0
                  if (value.length > 0 && value[0] === '0') {
                    value = value.substring(1); // Remove leading 0
                    setError('User ID must start with 1-9');
                    if (value.length > 0 && value[0] !== '0') {
                      setUserIdInput(value);
                    } else {
                      setUserIdInput('');
                    }
                    return;
                  }
                  
                  // Limit to 4 digits
                  if (value.length > 4) {
                    value = value.substring(0, 4);
                  }
                  
                  setUserIdInput(value);
                  setError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Backspace') {
                    e.preventDefault();
                    handleBackspace();
                  } else if (e.key === 'Enter' && userIdInput.length === 4) {
                    e.preventDefault();
                    if (authMode === 'login') handleLogin();
                    else handleSignup();
                  }
                }}
                className="absolute opacity-0 pointer-events-none w-0 h-0"
                autoFocus
                tabIndex={-1}
              />

              {/* Number pad */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '←', '✓'].map((item) => {
                  if (item === '←') {
                    return (
                      <Button
                        key={item}
                        variant="outline"
                        size="lg"
                        onClick={handleBackspace}
                        disabled={userIdInput.length === 0 || isLoading}
                        className="h-14 text-xl"
                      >
                        ←
                      </Button>
                    );
                  } else if (item === '✓') {
                    return (
                      <Button
                        key={item}
                        variant="landing"
                        size="lg"
                        onClick={authMode === 'login' ? handleLogin : handleSignup}
                        disabled={userIdInput.length !== 4 || isLoading}
                        className="h-14 text-xl"
                      >
                        {isLoading ? '...' : authMode === 'login' ? 'Login' : 'Sign Up'}
                      </Button>
                    );
                  } else {
                    const isDisabled = item === '0' && userIdInput.length === 0;
                    return (
                      <Button
                        key={item}
                        variant="outline"
                        size="lg"
                        onClick={() => handleUserIdInput(item)}
                        disabled={isDisabled || isLoading}
                        className="h-14 text-xl font-semibold"
                      >
                        {item}
                      </Button>
                    );
                  }
                })}
              </div>

              {/* Error message with switch button */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                    className="mt-4 text-center"
                  >
                    <p className="text-xs text-muted-foreground mb-3 uppercase tracking-widest">
                      {error}
                    </p>
                    {/* Show switch button if error suggests it */}
                    {(error.toLowerCase().includes('does not exist') || 
                      error.toLowerCase().includes('already exists')) && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.1 }}
                      >
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={authMode === 'login' ? switchToSignup : switchToLogin}
                          className="text-muted-foreground text-xs"
                        >
                          {authMode === 'login' 
                            ? 'Switch to Sign Up →' 
                            : 'Switch to Login →'}
                        </Button>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Switch to signup/login link (always shown when no error) */}
              {!error && (
                <div className="mt-6 text-center">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={authMode === 'login' ? switchToSignup : switchToLogin}
                    className="text-muted-foreground"
                  >
                    {authMode === 'login' 
                      ? "Don't have an account? Sign Up" 
                      : 'Already have an account? Login'}
                  </Button>
                </div>
              )}

              {/* Back button */}
              <div className="mt-4 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={backToInitial}
                  className="text-muted-foreground"
                >
                  ← Back
                </Button>
              </div>
              </motion.div>
            )}

            {/* Confirmation screen (after successful login/signup) */}
            {authMode === 'confirmed' && confirmedUserId && (
              <motion.div
                key="confirmed-card"
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="bg-card rounded-2xl shadow-elevated p-8 max-w-md mx-auto w-full text-center"
              >
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                  className="mb-6"
                >
                  <h2 className="text-2xl font-semibold mb-4 text-card-foreground">
                    {wasSignup ? 'Account Created!' : 'Welcome Back!'}
                  </h2>
                  <p className="text-muted-foreground mb-6 text-sm uppercase tracking-widest">
                    Your User ID
                  </p>
                  <div className="flex justify-center gap-3 mb-6">
                    {confirmedUserId.split('').map((digit, index) => (
                      <motion.div
                        key={index}
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.3 + index * 0.1, duration: 0.3 }}
                        className="w-16 h-16 rounded-full border-2 border-primary bg-primary/10 text-primary flex items-center justify-center text-3xl font-semibold"
                      >
                        {digit}
                      </motion.div>
                    ))}
                  </div>
                  <Button
                    variant="landing"
                    size="lg"
                    onClick={handleContinue}
                    className="min-w-[200px]"
                  >
                    Continue
                  </Button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
          className="text-muted-foreground text-sm mt-12 uppercase tracking-widest"
        >
          4 Questions • Personalized Recommendations • Endless Entertainment
        </motion.p>
      </motion.div>
    </div>
  );
}
