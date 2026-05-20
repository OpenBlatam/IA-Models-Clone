import { useState, useCallback } from 'react';

interface RetryOptions {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  onRetry?: (attempt: number) => void;
  onSuccess?: () => void;
  onFailure?: (error: Error) => void;
}

export const useRetry = <T,>(
  asyncFn: () => Promise<T>,
  options: RetryOptions = {}
) => {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    backoffMultiplier = 2,
    onRetry,
    onSuccess,
    onFailure,
  } = options;

  const [isRetrying, setIsRetrying] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const delay = useCallback(
    (ms: number) => new Promise((resolve) => setTimeout(resolve, ms)),
    []
  );

  const calculateDelay = useCallback(
    (attemptNumber: number): number => {
      const calculated = initialDelay * Math.pow(backoffMultiplier, attemptNumber);
      return Math.min(calculated, maxDelay);
    },
    [initialDelay, backoffMultiplier, maxDelay]
  );

  const execute = useCallback(async (): Promise<T | null> => {
    setIsRetrying(true);
    setError(null);
    let lastError: Error | null = null;

    for (let i = 0; i <= maxRetries; i++) {
      setAttempt(i);
      
      if (i > 0) {
        onRetry?.(i);
        const waitTime = calculateDelay(i - 1);
        await delay(waitTime);
      }

      try {
        const result = await asyncFn();
        setIsRetrying(false);
        setAttempt(0);
        onSuccess?.();
        return result;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        setError(lastError);

        if (i === maxRetries) {
          setIsRetrying(false);
          onFailure?.(lastError);
          return null;
        }
      }
    }

    setIsRetrying(false);
    return null;
  }, [asyncFn, maxRetries, calculateDelay, delay, onRetry, onSuccess, onFailure]);

  const reset = useCallback(() => {
    setIsRetrying(false);
    setAttempt(0);
    setError(null);
  }, []);

  return {
    execute,
    isRetrying,
    attempt,
    error,
    reset,
  };
};

