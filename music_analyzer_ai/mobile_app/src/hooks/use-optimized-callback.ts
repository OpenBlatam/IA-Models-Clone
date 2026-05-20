import { useCallback, useRef } from 'react';

/**
 * Optimized callback hook that maintains referential equality
 * while allowing dependencies to change
 */
export function useOptimizedCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  deps: React.DependencyList
): T {
  const callbackRef = useRef(callback);

  callbackRef.current = callback;

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps
  );
}

/**
 * Memoized callback that only updates when dependencies change
 * Useful for event handlers that need stable references
 */
export function useStableCallback<T extends (...args: unknown[]) => unknown>(
  callback: T
): T {
  const callbackRef = useRef(callback);

  callbackRef.current = callback;

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  ) as T;
}

