/**
 * usePrevious Hook
 * Hook to get the previous value of a prop or state
 */

import { useEffect, useRef } from 'react';

/**
 * Returns the previous value of the given value
 * @param value - The value to track
 * @returns The previous value
 */
export const usePrevious = <T,>(value: T): T | undefined => {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
};


