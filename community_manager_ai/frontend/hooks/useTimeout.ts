/**
 * useTimeout Hook
 * Hook for running a function after a delay
 */

import { useEffect, useRef } from 'react';

/**
 * Hook to run a function after a specified delay
 * @param callback - Function to execute
 * @param delay - Delay in milliseconds (null to cancel)
 */
export const useTimeout = (callback: () => void, delay: number | null) => {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const id = setTimeout(() => {
      savedCallback.current?.();
    }, delay);

    return () => clearTimeout(id);
  }, [delay]);
};


