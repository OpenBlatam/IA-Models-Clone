/**
 * Custom hook for deep comparison
 * 
 * Provides memoization based on deep equality
 */

import { useRef, useEffect } from "react";
import { deepEqual } from "../utils/comparison/diff";

/**
 * Custom hook for deep comparison
 * 
 * @param value - Value to compare
 * @param callback - Callback to call when value changes
 * 
 * @example
 * ```typescript
 * useDeepCompare(agent, (prev, current) => {
 *   console.log("Agent changed:", prev, current);
 * });
 * ```
 */
export function useDeepCompare<T>(
  value: T,
  callback: (prev: T | undefined, current: T) => void
): void {
  const ref = useRef<T>();

  useEffect(() => {
    if (!deepEqual(ref.current, value)) {
      callback(ref.current, value);
      ref.current = value;
    }
  }, [value, callback]);
}

/**
 * Custom hook that returns previous value if current is deeply equal
 */
export function useDeepCompareMemo<T>(value: T): T {
  const ref = useRef<T>(value);

  if (!deepEqual(ref.current, value)) {
    ref.current = value;
  }

  return ref.current;
}




