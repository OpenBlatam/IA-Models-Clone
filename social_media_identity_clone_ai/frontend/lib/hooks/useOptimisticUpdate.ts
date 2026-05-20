import { useState, useCallback } from 'react';

interface UseOptimisticUpdateOptions<T> {
  initialData: T;
  onUpdate: (data: T) => Promise<T>;
  onError?: (error: unknown) => void;
}

export const useOptimisticUpdate = <T,>({ initialData, onUpdate, onError }: UseOptimisticUpdateOptions<T>) => {
  const [data, setData] = useState<T>(initialData);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const update = useCallback(
    async (optimisticData: T) => {
      setData(optimisticData);
      setIsUpdating(true);
      setError(null);

      try {
        const result = await onUpdate(optimisticData);
        setData(result);
      } catch (err) {
        setData(initialData);
        setError(err);
        if (onError) {
          onError(err);
        }
        throw err;
      } finally {
        setIsUpdating(false);
      }
    },
    [initialData, onUpdate, onError]
  );

  const reset = useCallback(() => {
    setData(initialData);
    setError(null);
    setIsUpdating(false);
  }, [initialData]);

  return {
    data,
    isUpdating,
    error,
    update,
    reset,
  };
};



