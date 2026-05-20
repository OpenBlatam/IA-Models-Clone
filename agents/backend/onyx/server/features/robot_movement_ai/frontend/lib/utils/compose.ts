/**
 * Function composition utilities
 */

/**
 * Compose multiple functions from right to left
 */
export function compose<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
  return (arg: T) => fns.reduceRight((acc, fn) => fn(acc), arg);
}

/**
 * Compose multiple functions from left to right (pipe)
 */
export function pipe<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
  return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

/**
 * Create a function that applies multiple functions and returns the first truthy result
 */
export function firstTruthy<T, R>(
  ...fns: Array<(arg: T) => R | null | undefined>
): (arg: T) => R | null | undefined {
  return (arg: T) => {
    for (const fn of fns) {
      const result = fn(arg);
      if (result) return result;
    }
    return null;
  };
}

/**
 * Create a function that applies multiple functions and returns all results
 */
export function allResults<T, R>(
  ...fns: Array<(arg: T) => R>
): (arg: T) => R[] {
  return (arg: T) => fns.map((fn) => fn(arg));
}

/**
 * Create a memoized function
 */
export function memoize<T extends (...args: any[]) => any>(
  fn: T,
  keyFn?: (...args: Parameters<T>) => string
): T {
  const cache = new Map<string, ReturnType<T>>();

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

/**
 * Create a function that only runs once
 */
export function once<T extends (...args: any[]) => any>(fn: T): T {
  let called = false;
  let result: ReturnType<T>;

  return ((...args: Parameters<T>): ReturnType<T> => {
    if (!called) {
      called = true;
      result = fn(...args);
    }
    return result!;
  }) as T;
}

/**
 * Create a function that caches results based on arguments
 */
export function cacheByArgs<T extends (...args: any[]) => any>(
  fn: T,
  maxSize: number = 100
): T {
  const cache = new Map<string, ReturnType<T>>();
  const keys: string[] = [];

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);

    if (cache.size >= maxSize) {
      const oldestKey = keys.shift();
      if (oldestKey) {
        cache.delete(oldestKey);
      }
    }

    cache.set(key, result);
    keys.push(key);

    return result;
  }) as T;
}



