/**
 * Performance optimization utilities
 */

/**
 * Batch state updates
 */
export function batchUpdates(updates: Array<() => void>): () => void {
  return () => {
    updates.forEach((update) => update());
  };
}

/**
 * Throttle function calls
 */
export function throttleFn<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): T {
  let inThrottle: boolean;

  return ((...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  }) as T;
}

/**
 * Debounce function calls
 */
export function debounceFn<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): T {
  let timeoutId: NodeJS.Timeout | null = null;

  return ((...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
      timeoutId = null;
    }, delay);
  }) as T;
}

/**
 * Request idle callback wrapper
 */
export function requestIdleCallback(
  callback: () => void,
  options?: { timeout?: number }
): number {
  if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
    return window.requestIdleCallback(callback, options);
  }
  // Fallback
  return setTimeout(callback, 1) as unknown as number;
}

/**
 * Cancel idle callback
 */
export function cancelIdleCallback(id: number): void {
  if (typeof window !== 'undefined' && 'cancelIdleCallback' in window) {
    window.cancelIdleCallback(id);
  } else {
    clearTimeout(id as unknown as NodeJS.Timeout);
  }
}

/**
 * Lazy initialize value
 */
export function lazyInit<T>(initFn: () => T): () => T {
  let value: T | undefined;
  let initialized = false;

  return () => {
    if (!initialized) {
      value = initFn();
      initialized = true;
    }
    return value!;
  };
}

/**
 * Create virtual list helper
 */
export function createVirtualList<T>(
  items: T[],
  itemHeight: number,
  containerHeight: number
) {
  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const totalHeight = items.length * itemHeight;

  return {
    getVisibleItems: (scrollTop: number) => {
      const startIndex = Math.floor(scrollTop / itemHeight);
      const endIndex = Math.min(startIndex + visibleCount + 1, items.length);
      return {
        items: items.slice(startIndex, endIndex),
        startIndex,
        offsetY: startIndex * itemHeight,
      };
    },
    totalHeight,
    itemHeight,
  };
}



