import { useState, useCallback } from 'react';

interface UseAsyncStateOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

interface UseAsyncStateReturn<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  execute: (asyncFn: () => Promise<T>) => Promise<void>;
  reset: () => void;
}

export const useAsyncState = <T,>({
  onSuccess,
  onError,
}: UseAsyncStateOptions<T> = {}): UseAsyncStateReturn<T> => {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const execute = useCallback(
    async (asyncFn: () => Promise<T>): Promise<void> => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await asyncFn();
        setData(result);
        onSuccess?.(result);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        onError?.(error);
      } finally {
        setIsLoading(false);
      }
    },
    [onSuccess, onError]
  );

  const reset = useCallback((): void => {
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    data,
    error,
    isLoading,
    execute,
    reset,
  };
};

