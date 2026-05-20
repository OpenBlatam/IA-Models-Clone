/**
 * Performance utilities
 */

/**
 * Measure function execution time
 */
export const measurePerformance = <T,>(
  fn: () => T,
  label?: string
): T => {
  const start = performance.now();
  const result = fn();
  const end = performance.now();
  const duration = end - start;

  if (__DEV__ && label) {
    console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
  }

  return result;
};

/**
 * Measure async function execution time
 */
export const measureAsyncPerformance = async <T,>(
  fn: () => Promise<T>,
  label?: string
): Promise<T> => {
  const start = performance.now();
  const result = await fn();
  const end = performance.now();
  const duration = end - start;

  if (__DEV__ && label) {
    console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
  }

  return result;
};

/**
 * Debounce function
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
      func(...args);
    }, wait);
  };
};

/**
 * Throttle function
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
};

/**
 * Memoize function
 */
export const memoize = <T extends (...args: any[]) => any>(
  fn: T,
  getKey?: (...args: Parameters<T>) => string
): T => {
  const cache = new Map<string, ReturnType<T>>();

  return ((...args: Parameters<T>) => {
    const key = getKey ? getKey(...args) : JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
};

/**
 * Batch function calls
 */
export const batch = <T,>(
  items: T[],
  batchSize: number,
  processor: (batch: T[]) => void
): void => {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    processor(batch);
  }
};

