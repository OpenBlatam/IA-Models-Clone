/**
 * Custom hook for deep compare memo.
 * Memoizes value with deep comparison.
 */

import { useMemo, useRef } from 'react';
import { isEqual } from '../utils/object';

/**
 * Custom hook for deep compare memo.
 * Memoizes value with deep comparison.
 *
 * @param factory - Factory function
 * @param deps - Dependency array
 * @returns Memoized value
 */
export function useDeepCompareMemo<T>(
  factory: () => T,
  deps: React.DependencyList
): T {
  const ref = useRef<React.DependencyList>();
  const valueRef = useRef<T>();

  if (!ref.current || !isEqual(ref.current, deps)) {
    ref.current = deps;
    valueRef.current = factory();
  }

  return useMemo(() => valueRef.current as T, [ref.current]);
}

