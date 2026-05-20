import { useState, useEffect, useRef } from 'react';
import { AppState, AppStateStatus } from 'react-native';

interface UseIdleOptions {
  timeout?: number;
  initialState?: boolean;
}

/**
 * Hook to detect user idle state
 * Returns true when user is idle
 * Uses app state and manual reset
 */
export function useIdle({
  timeout = 30000,
  initialState = false,
}: UseIdleOptions = {}) {
  const [isIdle, setIsIdle] = useState(initialState);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastActivityRef = useRef<number>(Date.now());

  const resetIdle = () => {
    setIsIdle(false);
    lastActivityRef.current = Date.now();

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      setIsIdle(true);
    }, timeout);
  };

  useEffect(() => {
    // Set initial timeout
    timeoutRef.current = setTimeout(() => {
      setIsIdle(true);
    }, timeout);

    const subscription = AppState.addEventListener('change', (nextAppState: AppStateStatus) => {
      if (nextAppState === 'active') {
        resetIdle();
      }
    });

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      subscription.remove();
    };
  }, [timeout]);

  return {
    isIdle,
    resetIdle,
  };
}

