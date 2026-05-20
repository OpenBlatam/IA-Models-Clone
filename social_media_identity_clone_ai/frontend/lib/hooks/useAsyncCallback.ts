import { useCallback, useState, useRef } from 'react';

interface UseAsyncCallbackOptions {
  onSuccess?: (result: unknown) => void;
  onError?: (error: Error) => void;
}

export const useAsyncCallback = <T extends (...args: unknown[]) => Promise<unknown>>(
  asyncFunction: T,
  options: UseAsyncCallbackOptions = {}
): [T, boolean, Error | null] => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const mountedRef = useRef(true);

  const callback = useCallback(
    ((...args: Parameters<T>) => {
      setLoading(true);
      setError(null);

      return asyncFunction(...args)
        .then((result) => {
          if (mountedRef.current) {
            setLoading(false);
            if (options.onSuccess) {
              options.onSuccess(result);
            }
          }
          return result;
        })
        .catch((err) => {
          if (mountedRef.current) {
            const error = err instanceof Error ? err : new Error(String(err));
            setError(error);
            setLoading(false);
            if (options.onError) {
              options.onError(error);
            }
          }
          throw err;
        });
    }) as T,
    [asyncFunction, options.onSuccess, options.onError]
  );

  return [callback, loading, error];
};



