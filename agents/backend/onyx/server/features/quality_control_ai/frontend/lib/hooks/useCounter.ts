import { useState, useCallback } from 'react';

interface UseCounterOptions {
  initialValue?: number;
  min?: number;
  max?: number;
  step?: number;
}

export const useCounter = ({
  initialValue = 0,
  min,
  max,
  step = 1,
}: UseCounterOptions = {}) => {
  const [count, setCount] = useState(initialValue);

  const increment = useCallback((): void => {
    setCount((prev) => {
      const next = prev + step;
      return max !== undefined ? Math.min(next, max) : next;
    });
  }, [step, max]);

  const decrement = useCallback((): void => {
    setCount((prev) => {
      const next = prev - step;
      return min !== undefined ? Math.max(next, min) : next;
    });
  }, [step, min]);

  const reset = useCallback((): void => {
    setCount(initialValue);
  }, [initialValue]);

  const setValue = useCallback((value: number): void => {
    let newValue = value;
    if (min !== undefined) {
      newValue = Math.max(newValue, min);
    }
    if (max !== undefined) {
      newValue = Math.min(newValue, max);
    }
    setCount(newValue);
  }, [min, max]);

  return {
    count,
    increment,
    decrement,
    reset,
    setValue,
  };
};

