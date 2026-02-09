import { useState, useEffect, useCallback } from 'react';

interface UseQueryOptions {
  enabled?: boolean;
  refetchOnMount?: boolean;
  refetchOnWindowFocus?: boolean;
  staleTime?: number;
}

export const useQuery = <T,>(
  queryKey: string,
  queryFn: () => Promise<T>,
  options: UseQueryOptions = {}
) => {
  const {
    enabled = true,
    refetchOnMount = true,
    refetchOnWindowFocus = false,
    staleTime = 0,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<number>(0);

  const fetchData = useCallback(async () => {
    if (!enabled) return;

    const now = Date.now();
    if (staleTime > 0 && now - lastFetchTime < staleTime) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await queryFn();
      setData(result);
      setLastFetchTime(now);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
    } finally {
      setIsLoading(false);
    }
  }, [queryFn, enabled, staleTime, lastFetchTime]);

  useEffect(() => {
    if (refetchOnMount) {
      fetchData();
    }
  }, [refetchOnMount, fetchData]);

  const refetch = useCallback(() => {
    return fetchData();
  }, [fetchData]);

  return {
    data,
    isLoading,
    error,
    refetch,
    isStale: staleTime > 0 ? Date.now() - lastFetchTime > staleTime : false,
  };
};

