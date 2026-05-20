import { useState, useCallback } from 'react';
import { useErrorHandler } from './useErrorHandler';

export const useSafeAsync = <T,>() => {
  const [isLoading, setIsLoading] = useState(false);
  const { handleError } = useErrorHandler();

  const execute = useCallback(
    async (asyncFn: () => Promise<T>): Promise<T | null> => {
      setIsLoading(true);
      try {
        const result = await asyncFn();
        return result;
      } catch (error) {
        handleError(error);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [handleError]
  );

  return { execute, isLoading };
};

