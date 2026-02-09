import { useCallback, useRef, useEffect } from 'react';

export const useMemoizedCallback = <T extends (...args: any[]) => any>(
  callback: T,
  deps: React.DependencyList
): T => {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback, ...deps]);

  return useCallback(
    ((...args: any[]) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );
};

