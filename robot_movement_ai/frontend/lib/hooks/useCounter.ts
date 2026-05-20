import { useState, useCallback } from 'react';

export function useCounter(initialValue: number = 0, min?: number, max?: number) {
  const [count, setCount] = useState(initialValue);

  const increment = useCallback(() => {
    setCount((prev) => {
      const next = prev + 1;
      return max !== undefined && next > max ? prev : next;
    });
  }, [max]);

  const decrement = useCallback(() => {
    setCount((prev) => {
      const next = prev - 1;
      return min !== undefined && next < min ? prev : next;
    });
  }, [min]);

  const reset = useCallback(() => {
    setCount(initialValue);
  }, [initialValue]);

  const set = useCallback((value: number) => {
    setCount((prev) => {
      if (min !== undefined && value < min) return prev;
      if (max !== undefined && value > max) return prev;
      return value;
    });
  }, [min, max]);

  return {
    count,
    increment,
    decrement,
    reset,
    set,
  };
}



