import { useState, useEffect, useRef } from 'react';

interface UseCountdownOptions {
  initialSeconds: number;
  onComplete?: () => void;
  autoStart?: boolean;
}

/**
 * Hook for countdown timer
 * Counts down from initial seconds
 */
export function useCountdown({
  initialSeconds,
  onComplete,
  autoStart = false,
}: UseCountdownOptions) {
  const [seconds, setSeconds] = useState(initialSeconds);
  const [isRunning, setIsRunning] = useState(autoStart);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (isRunning && seconds > 0) {
      intervalRef.current = setInterval(() => {
        setSeconds((prev) => {
          if (prev <= 1) {
            setIsRunning(false);
            onComplete?.();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, seconds, onComplete]);

  const start = () => {
    setIsRunning(true);
  };

  const pause = () => {
    setIsRunning(false);
  };

  const reset = () => {
    setIsRunning(false);
    setSeconds(initialSeconds);
  };

  const restart = () => {
    setSeconds(initialSeconds);
    setIsRunning(true);
  };

  return {
    seconds,
    isRunning,
    start,
    pause,
    reset,
    restart,
  };
}

