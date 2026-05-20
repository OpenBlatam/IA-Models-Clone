/**
 * Custom hook for async operations.
 * Provides loading, error, and data state management for async functions.
 */

import { useState, useCallback } from 'react';

/**
 * Options for useAsync hook.
 */
export interface UseAsyncOptions<T> {
  immediate?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

/**
 * Return type for useAsync hook.
 */
export interface UseAsyncReturn<T> {
  execute: (...args: unknown[]) => Promise<T | undefined>;
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
  reset: () => void;
}

/**
 * Custom hook for managing async operations.
 * Provides loading, error, and data state for async functions.
 *
 * @param asyncFunction - Async function to execute
 * @param options - Hook options
 * @returns Async operation state and execute function
 */
export function useAsync<T>(
  asyncFunction: (...args: unknown[]) => Promise<T>,
  options: UseAsyncOptions<T> = {}
): UseAsyncReturn<T> {
  const { immediate = false, onSuccess, onError } = options;

  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(immediate);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isError, setIsError] = useState(false);

  /**
   * Executes the async function.
   */
  const execute = useCallback(
    async (...args: unknown[]): Promise<T | undefined> => {
      setIsLoading(true);
      setError(null);
      setIsSuccess(false);
      setIsError(false);

      try {
        const result = await asyncFunction(...args);
        setData(result);
        setIsSuccess(true);
        onSuccess?.(result);
        return result;
      } catch (err) {
        const errorInstance =
          err instanceof Error ? err : new Error(String(err));
        setError(errorInstance);
        setIsError(true);
        onError?.(errorInstance);
        return undefined;
      } finally {
        setIsLoading(false);
      }
    },
    [asyncFunction, onSuccess, onError]
  );

  /**
   * Resets the async state.
   */
  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setIsLoading(false);
    setIsSuccess(false);
    setIsError(false);
  }, []);

  // Execute immediately if requested
  if (immediate && !isLoading && !data && !error) {
    execute();
  }

  return {
    execute,
    data,
    error,
    isLoading,
    isSuccess,
    isError,
    reset,
  };
}

