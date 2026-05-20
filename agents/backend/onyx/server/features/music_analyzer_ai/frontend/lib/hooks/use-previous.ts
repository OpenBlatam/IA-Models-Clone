/**
 * Custom hook for tracking previous value.
 * Useful for comparing current and previous values.
 */

import { useRef, useEffect } from 'react';

/**
 * Custom hook that returns the previous value of a variable.
 * Useful for comparing current and previous values in effects.
 *
 * @param value - Value to track
 * @returns Previous value
 */
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

