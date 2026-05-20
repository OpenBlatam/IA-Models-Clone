/**
 * Custom hook for safe async actions with error handling.
 * Provides loading state, error handling, and success callbacks.
 */

import { useState, useCallback } from 'react';
import { getErrorMessage } from '@/lib/errors';

/**
 * Options for safe action hook.
 */
export interface UseSafeActionOptions<TData, TVariables> {
  action: (variables: TVariables) => Promise<TData>;
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (error: Error, variables: TVariables) => void;
  throwOnError?: boolean;
}

/**
 * Return type for useSafeAction hook.
 */
export interface UseSafeActionReturn<TData, TVariables> {
  execute: (variables: TVariables) => Promise<TData | undefined>;
  isLoading: boolean;
  error: Error | null;
  data: TData | null;
  reset: () => void;
}

/**
 * Custom hook for safe async actions with error handling.
 * @param options - Action options
 * @returns Safe action state and execute function
 */
export function useSafeAction<TData, TVariables = void>(
  options: UseSafeActionOptions<TData, TVariables>
): UseSafeActionReturn<TData, TVariables> {
  const { action, onSuccess, onError, throwOnError = false } = options;

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<TData | null>(null);

  /**
   * Executes the action safely with error handling.
   */
  const execute = useCallback(
    async (variables: TVariables): Promise<TData | undefined> => {
      setIsLoading(true);
      setError(null);

      try {
        const result = await action(variables);
        setData(result);
        onSuccess?.(result, variables);
        return result;
      } catch (err) {
        const errorInstance =
          err instanceof Error ? err : new Error(getErrorMessage(err));
        setError(errorInstance);
        onError?.(errorInstance, variables);

        if (throwOnError) {
          throw errorInstance;
        }

        return undefined;
      } finally {
        setIsLoading(false);
      }
    },
    [action, onSuccess, onError, throwOnError]
  );

  /**
   * Resets the action state.
   */
  const reset = useCallback(() => {
    setError(null);
    setData(null);
    setIsLoading(false);
  }, []);

  return {
    execute,
    isLoading,
    error,
    data,
    reset,
  };
}

