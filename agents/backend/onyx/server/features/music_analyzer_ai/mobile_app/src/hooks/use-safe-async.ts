import { useState, useCallback, useRef, useEffect } from 'react';
import type { AsyncState } from '../types';

interface UseSafeAsyncOptions {
  onSuccess?: (data: unknown) => void;
  onError?: (error: Error) => void;
}

/**
 * Safe async hook that prevents state updates on unmounted components
 * Follows React best practices for async operations
 */
export function useSafeAsync<T>(
  asyncFunction: () => Promise<T>,
  options?: UseSafeAsyncOptions
): AsyncState<T> & { execute: () => Promise<void>; reset: () => void } {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    isLoading: false,
    error: null,
  });

  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const execute = useCallback(async () => {
    if (!isMountedRef.current) {
      return;
    }

    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const data = await asyncFunction();

      if (!isMountedRef.current) {
        return;
      }

      setState({
        data,
        isLoading: false,
        error: null,
      });

      if (options?.onSuccess) {
        options.onSuccess(data);
      }
    } catch (error) {
      if (!isMountedRef.current) {
        return;
      }

      const errorObj =
        error instanceof Error
          ? error
          : new Error('An unexpected error occurred');

      setState({
        data: null,
        isLoading: false,
        error: {
          message: errorObj.message,
        },
      });

      if (options?.onError) {
        options.onError(errorObj);
      }
    }
  }, [asyncFunction, options]);

  const reset = useCallback(() => {
    if (!isMountedRef.current) {
      return;
    }

    setState({
      data: null,
      isLoading: false,
      error: null,
    });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

