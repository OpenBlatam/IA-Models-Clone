import { useRef, useCallback, useEffect } from 'react';

export const useStableCallback = <T extends (...args: unknown[]) => unknown>(
  callback: T
): T => {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );
};

