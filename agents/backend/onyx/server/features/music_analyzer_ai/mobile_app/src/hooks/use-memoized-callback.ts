import { useCallback, useRef } from 'react';
import type { DependencyList } from 'react';

/**
 * Creates a memoized callback that only changes when dependencies change.
 * Unlike useCallback, this uses a ref-based comparison for dependencies.
 */
export function useMemoizedCallback<T extends (...args: never[]) => unknown>(
  callback: T,
  deps: DependencyList
): T {
  const callbackRef = useRef(callback);
  const depsRef = useRef(deps);

  // Update callback ref if dependencies changed
  const hasChanged = deps.some(
    (dep, index) => dep !== depsRef.current[index]
  );

  if (hasChanged) {
    callbackRef.current = callback;
    depsRef.current = deps;
  }

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );
}

