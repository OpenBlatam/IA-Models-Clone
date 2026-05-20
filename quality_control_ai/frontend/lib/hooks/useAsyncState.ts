import { useState, useCallback } from 'react';

interface UseAsyncStateOptions<T> {
  initialData?: T;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

export const useAsyncState = <T,>(options: UseAsyncStateOptions<T> = {}) => {
  const [data, setData] = useState<T | undefined>(options.initialData);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const execute = useCallback(
    async (asyncFn: () => Promise<T>): Promise<T | undefined> => {
      setIsLoading(true);
      setError(null);

      try {
        const result = await asyncFn();
        setData(result);
        options.onSuccess?.(result);
        return result;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        options.onError?.(error);
        return undefined;
      } finally {
        setIsLoading(false);
      }
    },
    [options]
  );

  const reset = useCallback(() => {
    setData(options.initialData);
    setError(null);
    setIsLoading(false);
  }, [options.initialData]);

  return {
    data,
    error,
    isLoading,
    execute,
    reset,
    setData,
    setError,
  };
};

