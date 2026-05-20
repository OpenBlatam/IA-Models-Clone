/**
 * Custom hook for debounced value.
 * Provides debounced value updates for performance optimization.
 */

import { useState, useEffect } from 'react';

/**
 * Options for useDebounceValue hook.
 */
export interface UseDebounceValueOptions {
  delay: number;
  leading?: boolean;
}

/**
 * Custom hook for debounced value.
 * Debounces value updates to reduce re-renders.
 *
 * @param value - Value to debounce
 * @param options - Debounce options
 * @returns Debounced value
 */
export function useDebounceValue<T>(
  value: T,
  options: UseDebounceValueOptions
): T {
  const { delay, leading = false } = options;
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    if (leading) {
      setDebouncedValue(value);
      return;
    }

    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay, leading]);

  return debouncedValue;
}

