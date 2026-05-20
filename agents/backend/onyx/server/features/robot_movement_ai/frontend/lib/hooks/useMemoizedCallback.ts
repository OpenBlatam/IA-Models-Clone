import { useCallback, useRef } from 'react';

/**
 * Memoized callback that only changes when dependencies change
 * Similar to useCallback but with better dependency tracking
 */
export function useMemoizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: React.DependencyList
): T {
  const callbackRef = useRef(callback);

  // Update callback ref when dependencies change
  const memoizedCallback = useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    deps
  );

  callbackRef.current = callback;

  return memoizedCallback;
}



