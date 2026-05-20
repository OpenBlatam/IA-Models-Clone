/**
 * Custom hook for memo with deep comparison.
 * Provides memoization with custom comparison function.
 */

import { useRef, useMemo } from 'react';

/**
 * Custom hook for memo with custom comparison.
 * Memoizes value with custom comparison function.
 *
 * @param value - Value to memoize
 * @param compareFn - Comparison function
 * @returns Memoized value
 */
export function useMemoCompare<T>(
  value: T,
  compareFn: (prev: T | undefined, next: T) => boolean
): T {
  const ref = useRef<T>();

  if (!ref.current || !compareFn(ref.current, value)) {
    ref.current = value;
  }

  return ref.current;
}

