import { useState, useCallback } from 'react';

export interface UseLoadingReturn {
  isLoading: boolean;
  startLoading: () => void;
  stopLoading: () => void;
  withLoading: <T>(fn: () => Promise<T>) => Promise<T | null>;
}

/**
 * Hook for loading state management
 */
export function useLoading(initialState: boolean = false): UseLoadingReturn {
  const [isLoading, setIsLoading] = useState(initialState);

  const startLoading = useCallback(() => {
    setIsLoading(true);
  }, []);

  const stopLoading = useCallback(() => {
    setIsLoading(false);
  }, []);

  const withLoading = useCallback(
    async <T,>(fn: () => Promise<T>): Promise<T | null> => {
      startLoading();
      try {
        return await fn();
      } catch (error) {
        console.error('Error in withLoading:', error);
        return null;
      } finally {
        stopLoading();
      }
    },
    [startLoading, stopLoading]
  );

  return {
    isLoading,
    startLoading,
    stopLoading,
    withLoading,
  };
}



