/**
 * Advanced memoization utilities
 * @module robot-3d-view/utils/memoization
 */

/**
 * Memoized function with cache
 */
export interface MemoizedFunction<T extends (...args: unknown[]) => unknown> {
  (...args: Parameters<T>): ReturnType<T>;
  clear: () => void;
  cache: Map<string, ReturnType<T>>;
}

/**
 * Memoizes a function with custom key generator
 * 
 * @param fn - Function to memoize
 * @param keyGenerator - Key generator function
 * @param maxSize - Maximum cache size
 * @returns Memoized function
 */
export function memoize<T extends (...args: unknown[]) => unknown>(
  fn: T,
  keyGenerator?: (...args: Parameters<T>) => string,
  maxSize = 100
): MemoizedFunction<T> {
  const cache = new Map<string, ReturnType<T>>();
  const keys: string[] = [];

  const memoized = (...args: Parameters<T>): ReturnType<T> => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);

    // Limit cache size
    if (cache.size >= maxSize) {
      const oldestKey = keys.shift();
      if (oldestKey) {
        cache.delete(oldestKey);
      }
    }

    cache.set(key, result);
    keys.push(key);

    return result;
  };

  memoized.clear = () => {
    cache.clear();
    keys.length = 0;
  };

  memoized.cache = cache;

  return memoized;
}

/**
 * Weak memoization using WeakMap
 */
export function weakMemoize<T extends (...args: unknown[]) => unknown>(
  fn: T
): T {
  const cache = new WeakMap<object, ReturnType<T>>();

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = args[0] as object;
    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

/**
 * Debounced memoization
 */
export function debouncedMemoize<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number,
  keyGenerator?: (...args: Parameters<T>) => string
): MemoizedFunction<T> {
  const cache = new Map<string, ReturnType<T>>();
  const timers = new Map<string, NodeJS.Timeout>();

  const memoized = (...args: Parameters<T>): ReturnType<T> => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    // Clear existing timer
    const existingTimer = timers.get(key);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Set new timer
    const timer = setTimeout(() => {
      const result = fn(...args);
      cache.set(key, result);
      timers.delete(key);
    }, delay);

    timers.set(key, timer);

    // Return undefined or previous value while waiting
    return cache.get(key) as ReturnType<T>;
  };

  memoized.clear = () => {
    cache.clear();
    timers.forEach((timer) => clearTimeout(timer));
    timers.clear();
  };

  memoized.cache = cache;

  return memoized;
}

/**
 * Memoizes async functions
 */
export function memoizeAsync<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  keyGenerator?: (...args: Parameters<T>) => string,
  maxSize = 100
): MemoizedFunction<T> {
  const cache = new Map<string, Promise<ReturnType<T>>>();
  const keys: string[] = [];

  const memoized = async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const promise = Promise.resolve(fn(...args)) as Promise<ReturnType<T>>;

    // Limit cache size
    if (cache.size >= maxSize) {
      const oldestKey = keys.shift();
      if (oldestKey) {
        cache.delete(oldestKey);
      }
    }

    cache.set(key, promise);
    keys.push(key);

    return promise;
  };

  memoized.clear = () => {
    cache.clear();
    keys.length = 0;
  };

  memoized.cache = cache as Map<string, ReturnType<T>>;

  return memoized;
}



