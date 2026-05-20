/**
 * Memoization utilities
 */

/**
 * Create memoized function with custom cache key
 */
export function memoizeWithKey<T extends (...args: any[]) => any>(
  fn: T,
  keyFn: (...args: Parameters<T>) => string
): T {
  const cache = new Map<string, ReturnType<T>>();

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = keyFn(...args);
    
    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

/**
 * Create weak memoized function (uses WeakMap)
 */
export function weakMemoize<T extends (arg: object) => any>(
  fn: T
): T {
  const cache = new WeakMap<object, ReturnType<T>>();

  return ((arg: object): ReturnType<T> => {
    if (cache.has(arg)) {
      return cache.get(arg)!;
    }

    const result = fn(arg);
    cache.set(arg, result);
    return result;
  }) as T;
}

/**
 * Create LRU cache
 */
export function createLRUCache<TKey, TValue>(maxSize: number = 100) {
  const cache = new Map<TKey, TValue>();
  const accessOrder: TKey[] = [];

  return {
    get: (key: TKey): TValue | undefined => {
      if (cache.has(key)) {
        // Move to end (most recently used)
        const index = accessOrder.indexOf(key);
        accessOrder.splice(index, 1);
        accessOrder.push(key);
        return cache.get(key);
      }
      return undefined;
    },

    set: (key: TKey, value: TValue): void => {
      if (cache.has(key)) {
        // Update existing
        cache.set(key, value);
        const index = accessOrder.indexOf(key);
        accessOrder.splice(index, 1);
        accessOrder.push(key);
      } else {
        // Add new
        if (cache.size >= maxSize) {
          // Remove least recently used
          const lruKey = accessOrder.shift()!;
          cache.delete(lruKey);
        }
        cache.set(key, value);
        accessOrder.push(key);
      }
    },

    clear: (): void => {
      cache.clear();
      accessOrder.length = 0;
    },

    size: (): number => cache.size,
  };
}

/**
 * Create memoized selector
 */
export function createSelector<TState, TResult>(
  selector: (state: TState) => TResult
): (state: TState) => TResult {
  let lastState: TState | null = null;
  let lastResult: TResult | null = null;

  return (state: TState): TResult => {
    if (state === lastState && lastResult !== null) {
      return lastResult;
    }

    lastState = state;
    lastResult = selector(state);
    return lastResult;
  };
}



