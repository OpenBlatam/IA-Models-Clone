import { useState, useCallback } from 'react';
import { retry, RetryOptions } from '@/lib/utils/retry';

export function useRetry<T>() {
  const [isRetrying, setIsRetrying] = useState(false);
  const [attempt, setAttempt] = useState(0);

  const executeWithRetry = useCallback(
    async (
      fn: () => Promise<T>,
      options: RetryOptions = {}
    ): Promise<T | null> => {
      setIsRetrying(true);
      setAttempt(0);

      try {
        const result = await retry(fn, {
          ...options,
          onRetry: (currentAttempt, error) => {
            setAttempt(currentAttempt);
            options.onRetry?.(currentAttempt, error);
          },
        });

        setIsRetrying(false);
        setAttempt(0);
        return result;
      } catch (error) {
        setIsRetrying(false);
        setAttempt(0);
        console.error('Retry failed:', error);
        return null;
      }
    },
    []
  );

  return {
    executeWithRetry,
    isRetrying,
    attempt,
  };
}



