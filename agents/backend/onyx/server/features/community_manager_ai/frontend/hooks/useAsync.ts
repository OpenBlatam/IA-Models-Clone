/**
 * useAsync Hook
 * Hook for handling async operations with loading, error, and data states
 */

import { useState, useCallback, useEffect, useRef } from 'react';

interface UseAsyncOptions<T> {
  immediate?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

interface UseAsyncState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

/**
 * Hook for managing async operations
 * @param asyncFunction - The async function to execute
 * @param options - Configuration options
 * @returns Object with execute function and state
 */
export const useAsync = <T,>(
  asyncFunction: () => Promise<T>,
  options: UseAsyncOptions<T> = {}
) => {
  const { immediate = false, onSuccess, onError } = options;
  const [state, setState] = useState<UseAsyncState<T>>({
    data: null,
    error: null,
    loading: false,
  });
  const mountedRef = useRef(true);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const execute = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const data = await asyncFunction();
      
      if (!mountedRef.current) return;

      setState({ data, error: null, loading: false });
      onSuccess?.(data);
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Unknown error');
      
      if (!mountedRef.current) return;

      setState((prev) => ({ ...prev, error: err, loading: false }));
      onError?.(err);
      throw err;
    }
  }, [asyncFunction, onSuccess, onError]);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [immediate, execute]);

  const reset = useCallback(() => {
    setState({ data: null, error: null, loading: false });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
};


