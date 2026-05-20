import { useState, useCallback } from 'react';
import { useToast } from './useToast';

interface UseAsyncOperationOptions {
  onSuccess?: (result: unknown) => void;
  onError?: (error: Error) => void;
  successMessage?: string;
  errorMessage?: string;
  suppressErrorToast?: boolean;
}

export const useAsyncOperation = <T,>(
  operation: () => Promise<T>,
  options: UseAsyncOperationOptions = {}
) => {
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const execute = useCallback(async (): Promise<T | null> => {
    setIsLoading(true);
    try {
      const result = await operation();
      if (options.successMessage) {
        toast.success(options.successMessage);
      }
      options.onSuccess?.(result);
      return result;
    } catch (error) {
      const message =
        options.errorMessage ||
        (error instanceof Error ? error.message : 'An error occurred');
      
      if (!options.suppressErrorToast) {
        toast.error(message);
      }
      
      options.onError?.(error instanceof Error ? error : new Error(message));
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [operation, options, toast]);

  return { execute, isLoading };
};
