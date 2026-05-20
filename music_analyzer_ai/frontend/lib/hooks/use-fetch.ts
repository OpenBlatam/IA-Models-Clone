/**
 * Custom hook for data fetching.
 * Provides convenient data fetching with loading, error, and data states.
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * Options for useFetch hook.
 */
export interface UseFetchOptions<T> {
  immediate?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  refetchInterval?: number;
}

/**
 * Return type for useFetch hook.
 */
export interface UseFetchReturn<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  reset: () => void;
}

/**
 * Custom hook for data fetching.
 * Provides convenient data fetching with state management.
 *
 * @param fetcher - Async function that returns data
 * @param options - Hook options
 * @returns Fetch state and controls
 */
export function useFetch<T>(
  fetcher: () => Promise<T>,
  options: UseFetchOptions<T> = {}
): UseFetchReturn<T> {
  const { immediate = true, onSuccess, onError, refetchInterval } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await fetcher();
      setData(result);
      onSuccess?.(result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      onError?.(error);
    } finally {
      setLoading(false);
    }
  }, [fetcher, onSuccess, onError]);

  useEffect(() => {
    if (immediate) {
      fetchData();
    }
  }, [immediate, fetchData]);

  useEffect(() => {
    if (refetchInterval && refetchInterval > 0) {
      const interval = setInterval(() => {
        fetchData();
      }, refetchInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [refetchInterval, fetchData]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    data,
    loading,
    error,
    refetch: fetchData,
    reset,
  };
}

