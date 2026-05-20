/**
 * Memoization utility functions.
 * Provides helper functions for function memoization.
 */

/**
 * Memoized function type.
 */
export type MemoizedFunction<T extends (...args: any[]) => any> = T & {
  clear: () => void;
  cache: Map<string, ReturnType<T>>;
};

/**
 * Simple memoization function.
 */
export function memoize<T extends (...args: any[]) => any>(
  fn: T,
  keyGenerator?: (...args: Parameters<T>) => string
): MemoizedFunction<T> {
  const cache = new Map<string, ReturnType<T>>();

  const memoized = ((...args: Parameters<T>) => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as MemoizedFunction<T>;

  memoized.clear = () => cache.clear();
  memoized.cache = cache;

  return memoized;
}

/**
 * Memoization with max cache size (LRU).
 */
export function memoizeLRU<T extends (...args: any[]) => any>(
  fn: T,
  maxSize: number = 100,
  keyGenerator?: (...args: Parameters<T>) => string
): MemoizedFunction<T> {
  const cache = new Map<string, ReturnType<T>>();

  const memoized = ((...args: Parameters<T>) => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      // Move to end (most recently used)
      const value = cache.get(key)!;
      cache.delete(key);
      cache.set(key, value);
      return value;
    }

    const result = fn(...args);

    if (cache.size >= maxSize) {
      // Remove least recently used (first item)
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }

    cache.set(key, result);
    return result;
  }) as MemoizedFunction<T>;

  memoized.clear = () => cache.clear();
  memoized.cache = cache;

  return memoized;
}

/**
 * Memoization with TTL (Time To Live).
 */
export function memoizeTTL<T extends (...args: any[]) => any>(
  fn: T,
  ttl: number = 60000,
  keyGenerator?: (...args: Parameters<T>) => string
): MemoizedFunction<T> {
  const cache = new Map<string, { value: ReturnType<T>; expiresAt: number }>();

  const memoized = ((...args: Parameters<T>) => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    const entry = cache.get(key);

    if (entry && Date.now() < entry.expiresAt) {
      return entry.value;
    }

    const result = fn(...args);
    cache.set(key, {
      value: result,
      expiresAt: Date.now() + ttl,
    });

    return result;
  }) as MemoizedFunction<T>;

  memoized.clear = () => cache.clear();
  memoized.cache = new Map();

  return memoized;
}

