/**
 * Custom hook for debouncing values.
 * Useful for search inputs and other scenarios where you want to delay updates.
 */

import { useEffect, useState } from 'react';

/**
 * Debounces a value, returning the debounced value after the delay.
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 500)
 * @returns Debounced value
 */
export function useDebounce<T>(value: T, delay: number = 500): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

