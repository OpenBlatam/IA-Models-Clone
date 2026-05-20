import { useMemo, useCallback, useRef } from 'react';
import type { DependencyList } from 'react';

/**
 * Memoize a value with custom equality function
 */
export function useMemoWithEquality<T>(
  factory: () => T,
  deps: DependencyList,
  equalityFn?: (prev: T, next: T) => boolean
): T {
  const prevValueRef = useRef<T>();
  const prevDepsRef = useRef<DependencyList>();

  return useMemo(() => {
    const hasChanged =
      !prevDepsRef.current ||
      !deps.every((dep, index) => dep === prevDepsRef.current?.[index]);

    if (hasChanged) {
      const newValue = factory();

      if (
        equalityFn &&
        prevValueRef.current !== undefined &&
        equalityFn(prevValueRef.current, newValue)
      ) {
        return prevValueRef.current;
      }

      prevValueRef.current = newValue;
      prevDepsRef.current = deps;
      return newValue;
    }

    return prevValueRef.current as T;
  }, [factory, deps, equalityFn]);
}

