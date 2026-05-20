import { useCallback, useRef } from 'react';

export function useMemoizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: React.DependencyList
): T {
  const callbackRef = useRef(callback);

  callbackRef.current = callback;

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    deps
  );
}

