import { useRef, useEffect } from 'react';

/**
 * Hook to track previous value of a variable
 * Useful for detecting changes and comparisons
 */
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

/**
 * Hook to check if value has changed from previous
 */
export function useHasChanged<T>(value: T): boolean {
  const prevValue = usePrevious(value);
  return prevValue !== undefined && prevValue !== value;
}

