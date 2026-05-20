/**
 * Custom hooks for memoized callbacks.
 * Provides optimized callback hooks to prevent unnecessary re-renders.
 */

import { useCallback, useMemo, type DependencyList } from 'react';

/**
 * Creates a memoized callback that only changes when dependencies change.
 * Useful for event handlers passed to child components.
 * @param callback - The callback function
 * @param deps - Dependency array
 * @returns Memoized callback
 */
export function useMemoizedCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  deps: DependencyList
): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useCallback(callback, deps) as T;
}

/**
 * Creates multiple memoized callbacks at once.
 * Useful when you need multiple related callbacks.
 * Note: This is a simplified version. For better performance, memoize each callback individually.
 * @param callbacks - Object with callback functions
 * @param deps - Dependency array
 * @returns Object with memoized callbacks
 */
export function useMemoizedCallbacks<T extends Record<string, (...args: unknown[]) => unknown>>(
  callbacks: T,
  deps: DependencyList
): T {
  // Use useMemo to create a stable object reference
  return useMemo(() => {
    const memoized: Partial<T> = {};
    for (const key in callbacks) {
      if (Object.prototype.hasOwnProperty.call(callbacks, key)) {
        // Store the callback function directly
        // The memoization happens at the component level
        memoized[key] = callbacks[key] as T[Extract<keyof T, string>];
      }
    }
    return memoized as T;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...Object.values(callbacks), ...deps]);
}

/**
 * Creates a stable reference to a value that only changes when dependencies change.
 * Similar to useMemo but with a more explicit API for object references.
 * @param factory - Factory function that returns the value
 * @param deps - Dependency array
 * @returns Stable reference to the value
 */
export function useStableValue<T>(factory: () => T, deps: DependencyList): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(factory, deps);
}

