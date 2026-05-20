import { useRef, useCallback } from 'react';

export const useMemoizedCallback = <T extends (...args: unknown[]) => unknown>(
  callback: T,
  deps: React.DependencyList
): T => {
  const callbackRef = useRef(callback);
  const depsRef = useRef(deps);

  // Check if deps changed
  const depsChanged =
    depsRef.current.length !== deps.length ||
    depsRef.current.some((dep, i) => dep !== deps[i]);

  if (depsChanged) {
    callbackRef.current = callback;
    depsRef.current = deps;
  }

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );
};

