/**
 * useInterval Hook
 * Hook for running a function at specified intervals
 */

import { useEffect, useRef } from 'react';

/**
 * Hook to run a function at specified intervals
 * @param callback - Function to execute
 * @param delay - Delay in milliseconds (null to pause)
 */
export const useInterval = (callback: () => void, delay: number | null) => {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const id = setInterval(() => {
      savedCallback.current?.();
    }, delay);

    return () => clearInterval(id);
  }, [delay]);
};


