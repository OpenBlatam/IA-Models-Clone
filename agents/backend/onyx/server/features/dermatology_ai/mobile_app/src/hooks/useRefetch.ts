import { useCallback, useRef } from 'react';

export const useRefetch = <T,>(
  fetchFunction: () => Promise<T>,
  onSuccess?: (data: T) => void,
  onError?: (error: Error) => void
) => {
  const isRefetching = useRef(false);

  const refetch = useCallback(async () => {
    if (isRefetching.current) return;

    try {
      isRefetching.current = true;
      const data = await fetchFunction();
      onSuccess?.(data);
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      onError?.(err);
      throw err;
    } finally {
      isRefetching.current = false;
    }
  }, [fetchFunction, onSuccess, onError]);

  return {
    refetch,
    isRefetching: isRefetching.current,
  };
};

