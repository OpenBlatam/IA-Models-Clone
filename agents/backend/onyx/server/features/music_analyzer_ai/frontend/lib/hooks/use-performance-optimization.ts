/**
 * Performance optimization hook.
 * Provides utilities for optimizing component performance.
 */

import { useMemo, useCallback, useRef, useEffect, type DependencyList } from 'react';

/**
 * Options for useMemoizedValue hook.
 */
export interface UseMemoizedValueOptions {
  /**
   * Custom equality function for comparing values.
   */
  equalityFn?: (a: unknown, b: unknown) => boolean;
}

/**
 * Memoizes a value with optional custom equality function.
 * More flexible than useMemo for complex comparison logic.
 *
 * @param factory - Function that returns the value
 * @param deps - Dependencies array
 * @param options - Options for memoization
 * @returns Memoized value
 *
 * @example
 * ```tsx
 * const memoizedValue = useMemoizedValue(
 *   () => expensiveCalculation(data),
 *   [data],
 *   { equalityFn: deepEqual }
 * );
 * ```
 */
export function useMemoizedValue<T>(
  factory: () => T,
  deps: DependencyList,
  options?: UseMemoizedValueOptions
): T {
  const prevDepsRef = useRef<DependencyList>([]);
  const prevValueRef = useRef<T | undefined>(undefined);

  return useMemo(() => {
    const hasChanged = options?.equalityFn
      ? !deps.every((dep, i) => options.equalityFn!(dep, prevDepsRef.current[i]))
      : !deps.every((dep, i) => Object.is(dep, prevDepsRef.current[i]));

    if (hasChanged || prevValueRef.current === undefined) {
      prevDepsRef.current = deps;
      prevValueRef.current = factory();
    }

    return prevValueRef.current;
  }, deps);
}

/**
 * Creates a stable callback reference that only changes when dependencies change.
 * Similar to useCallback but with better type inference.
 *
 * @param callback - Callback function
 * @param deps - Dependencies array
 * @returns Stable callback reference
 *
 * @example
 * ```tsx
 * const handleClick = useStableCallback((id: string) => {
 *   doSomething(id);
 * }, [dependency]);
 * ```
 */
export function useStableCallback<T extends (...args: never[]) => unknown>(
  callback: T,
  deps: DependencyList
): T {
  return useCallback(callback, deps) as T;
}

/**
 * Options for useDebouncedCallback hook.
 */
export interface UseDebouncedCallbackOptions {
  /**
   * Delay in milliseconds.
   */
  delay?: number;
  /**
   * Whether to call the callback immediately on first invocation.
   */
  leading?: boolean;
  /**
   * Whether to call the callback on trailing edge.
   */
  trailing?: boolean;
}

/**
 * Creates a debounced callback that delays execution.
 *
 * @param callback - Callback function
 * @param delay - Delay in milliseconds
 * @param deps - Dependencies array
 * @param options - Debounce options
 * @returns Debounced callback
 *
 * @example
 * ```tsx
 * const debouncedSearch = useDebouncedCallback(
 *   (query: string) => search(query),
 *   300,
 *   []
 * );
 * ```
 */
export function useDebouncedCallback<T extends (...args: never[]) => unknown>(
  callback: T,
  delay: number,
  deps: DependencyList,
  options?: UseDebouncedCallbackOptions
): T {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const debouncedCallback = useCallback(
    ((...args: Parameters<T>) => {
      const { leading = false, trailing = true } = options ?? {};

      if (leading && !timeoutRef.current) {
        callbackRef.current(...args);
      }

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        if (trailing) {
          callbackRef.current(...args);
        }
        timeoutRef.current = undefined;
      }, delay);
    }) as T,
    [delay, ...deps]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback;
}

/**
 * Options for useThrottledCallback hook.
 */
export interface UseThrottledCallbackOptions {
  /**
   * Whether to call the callback immediately on first invocation.
   */
  leading?: boolean;
  /**
   * Whether to call the callback on trailing edge.
   */
  trailing?: boolean;
}

/**
 * Creates a throttled callback that limits execution frequency.
 *
 * @param callback - Callback function
 * @param delay - Throttle delay in milliseconds
 * @param deps - Dependencies array
 * @param options - Throttle options
 * @returns Throttled callback
 *
 * @example
 * ```tsx
 * const throttledScroll = useThrottledCallback(
 *   () => handleScroll(),
 *   100,
 *   []
 * );
 * ```
 */
export function useThrottledCallback<T extends (...args: never[]) => unknown>(
  callback: T,
  delay: number,
  deps: DependencyList,
  options?: UseThrottledCallbackOptions
): T {
  const lastRunRef = useRef<number>(0);
  const timeoutRef = useRef<NodeJS.Timeout>();
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const throttledCallback = useCallback(
    ((...args: Parameters<T>) => {
      const now = Date.now();
      const { leading = true, trailing = true } = options ?? {};

      if (leading && now - lastRunRef.current >= delay) {
        lastRunRef.current = now;
        callbackRef.current(...args);
        return;
      }

      if (timeoutRef.current) {
        return;
      }

      if (trailing) {
        timeoutRef.current = setTimeout(() => {
          lastRunRef.current = Date.now();
          callbackRef.current(...args);
          timeoutRef.current = undefined;
        }, delay - (now - lastRunRef.current));
      }
    }) as T,
    [delay, ...deps]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return throttledCallback;
}

/**
 * Tracks component render count for performance debugging.
 *
 * @param componentName - Name of the component
 * @returns Render count
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const renderCount = useRenderCount('MyComponent');
 *   console.log(`Rendered ${renderCount} times`);
 *   return <div>...</div>;
 * }
 * ```
 */
export function useRenderCount(componentName?: string): number {
  const countRef = useRef(0);

  useEffect(() => {
    countRef.current += 1;
    if (componentName && typeof window !== 'undefined' && window.console) {
      console.log(`[${componentName}] Render count: ${countRef.current}`);
    }
  });

  return countRef.current;
}




