/**
 * Custom hook for timeout management.
 * Provides timeout functionality with automatic cleanup.
 */

import { useEffect, useRef } from 'react';

/**
 * Options for useTimeout hook.
 */
export interface UseTimeoutOptions {
  delay: number | null;
  immediate?: boolean;
}

/**
 * Custom hook for timeout management.
 * Runs a callback after a specified delay with automatic cleanup.
 *
 * @param callback - Function to call after delay
 * @param options - Hook options
 */
export function useTimeout(
  callback: () => void,
  options: UseTimeoutOptions
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
      return;
    }

    const timeoutId = setTimeout(() => {
      callbackRef.current();
    }, delay);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [delay, immediate]);
}

