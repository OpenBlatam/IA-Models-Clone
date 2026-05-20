/**
 * Custom hook for interval management.
 * Provides interval functionality with automatic cleanup.
 */

import { useEffect, useRef } from 'react';

/**
 * Options for useInterval hook.
 */
export interface UseIntervalOptions {
  delay: number | null;
  immediate?: boolean;
}

/**
 * Custom hook for interval management.
 * Runs a callback at specified intervals with automatic cleanup.
 *
 * @param callback - Function to call on each interval
 * @param options - Hook options
 */
export function useInterval(
  callback: () => void,
  options: UseIntervalOptions
): void {
  const { delay, immediate = false } = options;
  const callbackRef = useRef(callback);

  // Update callback ref when it changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) {
      return;
    }

    // Run immediately if requested
    if (immediate) {
      callbackRef.current();
    }

    const intervalId = setInterval(() => {
      callbackRef.current();
    }, delay);

    return () => {
      clearInterval(intervalId);
    };
  }, [delay, immediate]);
}

