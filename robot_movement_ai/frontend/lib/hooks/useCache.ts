import { useState, useCallback, useEffect } from 'react';
import { cache } from '@/lib/utils/cache';

export function useCache<T>(
  key: string,
  fetcher: () => Promise<T>,
  ttl: number = 60000
) {
  const [data, setData] = useState<T | null>(cache.get<T>(key));
  const [isLoading, setIsLoading] = useState(!data);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await fetcher();
      cache.set(key, result, ttl);
      setData(result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
    } finally {
      setIsLoading(false);
    }
  }, [key, fetcher, ttl]);

  useEffect(() => {
    if (!data) {
      fetchData();
    }
  }, [data, fetchData]);

  const invalidate = useCallback(() => {
    cache.delete(key);
    setData(null);
    fetchData();
  }, [key, fetchData]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchData,
    invalidate,
  };
}



