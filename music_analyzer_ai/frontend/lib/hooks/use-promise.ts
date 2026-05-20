/**
 * Custom hook for promise state management.
 * Provides reactive promise state handling.
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * Promise state.
 */
export interface PromiseState<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
}

/**
 * Custom hook for promise state management.
 * Provides reactive promise state handling.
 *
 * @param promiseFn - Promise function
 * @param deps - Dependency array
 * @returns Promise state
 */
export function usePromise<T>(
  promiseFn: () => Promise<T> | null,
  deps: React.DependencyList = []
): PromiseState<T> {
  const [state, setState] = useState<PromiseState<T>>({
    data: null,
    error: null,
    isLoading: true,
    isSuccess: false,
    isError: false,
  });

  const execute = useCallback(async () => {
    const promise = promiseFn();
    if (!promise) {
      setState({
        data: null,
        error: null,
        isLoading: false,
        isSuccess: false,
        isError: false,
      });
      return;
    }

    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const data = await promise;
      setState({
        data,
        error: null,
        isLoading: false,
        isSuccess: true,
        isError: false,
      });
    } catch (error) {
      setState({
        data: null,
        error: error as Error,
        isLoading: false,
        isSuccess: false,
        isError: true,
      });
    }
  }, [promiseFn, ...deps]);

  useEffect(() => {
    execute();
  }, [execute]);

  return state;
}

