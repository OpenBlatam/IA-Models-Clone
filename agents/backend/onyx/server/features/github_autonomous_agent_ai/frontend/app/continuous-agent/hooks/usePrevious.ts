/**
 * Custom hook to track previous value
 * 
 * Useful for comparing current and previous values in effects
 */

import { useRef, useEffect } from "react";

/**
 * Custom hook to track previous value
 * 
 * @param value - Value to track
 * @returns Previous value
 * 
 * @example
 * ```typescript
 * const [count, setCount] = useState(0);
 * const prevCount = usePrevious(count);
 * 
 * useEffect(() => {
 *   if (count > prevCount) {
 *     console.log("Count increased");
 *   }
 * }, [count, prevCount]);
 * ```
 */
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}




