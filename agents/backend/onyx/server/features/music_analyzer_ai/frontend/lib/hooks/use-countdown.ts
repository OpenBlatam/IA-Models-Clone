/**
 * Custom hook for countdown timer.
 * Provides countdown functionality with callbacks.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Options for useCountdown hook.
 */
export interface UseCountdownOptions {
  initialSeconds?: number;
  onComplete?: () => void;
  autoStart?: boolean;
}

/**
 * Return type for useCountdown hook.
 */
export interface UseCountdownReturn {
  seconds: number;
  isRunning: boolean;
  start: () => void;
  pause: () => void;
  reset: () => void;
  setSeconds: (seconds: number) => void;
}

/**
 * Custom hook for countdown timer.
 * Provides countdown functionality with controls.
 *
 * @param options - Hook options
 * @returns Countdown state and controls
 */
export function useCountdown(
  options: UseCountdownOptions = {}
): UseCountdownReturn {
  const {
    initialSeconds = 0,
    onComplete,
    autoStart = false,
  } = options;

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

  const start = useCallback(() => {
    setIsRunning(true);
  }, []);

  const pause = useCallback(() => {
    setIsRunning(false);
  }, []);

  const reset = useCallback(() => {
    setIsRunning(false);
    setSeconds(initialSeconds);
  }, [initialSeconds]);

  return {
    seconds,
    isRunning,
    start,
    pause,
    reset,
    setSeconds,
  };
}

