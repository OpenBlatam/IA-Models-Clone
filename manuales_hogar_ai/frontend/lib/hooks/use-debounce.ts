import { useState, useEffect } from 'react';
import { SEARCH } from '../constants';

export const useDebounce = <T,>(value: T, delay: number = SEARCH.DEBOUNCE_MS): T => {
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
};

