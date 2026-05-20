import { useState, useCallback } from 'react';
import { apiRequest, ApiRequestOptions } from '@/lib/utils/api-helpers';
import { handleApiError } from '@/lib/utils/error-handler';

export function useApi<T>() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const request = useCallback(
    async (endpoint: string, options?: ApiRequestOptions): Promise<T | null> => {
      setLoading(true);
      setError(null);

      try {
        const data = await apiRequest<T>(endpoint, options);
        setLoading(false);
        return data;
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Unknown error');
        setError(error);
        setLoading(false);
        handleApiError(error);
        return null;
      }
    },
    []
  );

  const reset = useCallback(() => {
    setError(null);
    setLoading(false);
  }, []);

  return {
    request,
    loading,
    error,
    reset,
  };
}



