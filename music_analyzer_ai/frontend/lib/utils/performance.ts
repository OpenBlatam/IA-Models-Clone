/**
 * Performance utility functions.
 * Provides utilities for performance optimization and monitoring.
 */

/**
 * Throttle function to limit how often a function can be called.
 * @param func - Function to throttle
 * @param limit - Time limit in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return function (this: unknown, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

/**
 * Request animation frame throttle.
 * Useful for scroll and resize handlers.
 * @param func - Function to throttle
 * @returns Throttled function using requestAnimationFrame
 */
export function rafThrottle<T extends (...args: unknown[]) => unknown>(
  func: T
): (...args: Parameters<T>) => void {
  let rafId: number | null = null;
  return function (this: unknown, ...args: Parameters<T>) {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
    }
    rafId = requestAnimationFrame(() => {
      func.apply(this, args);
      rafId = null;
    });
  };
}

/**
 * Checks if code is running in browser environment.
 * @returns True if running in browser
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined';
}

/**
 * Checks if code is running in production environment.
 * @returns True if running in production
 */
export function isProduction(): boolean {
  return process.env.NODE_ENV === 'production';
}

/**
 * Measures performance of a function execution.
 * @param name - Name for the performance mark
 * @param func - Function to measure
 * @returns Result of function execution
 */
export function measurePerformance<T>(
  name: string,
  func: () => T
): T {
  if (!isBrowser() || !isProduction()) {
    return func();
  }

  const startMark = `${name}-start`;
  const endMark = `${name}-end`;
  const measureName = `${name}-measure`;

  performance.mark(startMark);
  const result = func();
  performance.mark(endMark);
  performance.measure(measureName, startMark, endMark);

  const measure = performance.getEntriesByName(measureName)[0];
  if (measure) {
    console.log(`Performance [${name}]: ${measure.duration.toFixed(2)}ms`);
  }

  return result;
}

// Note: For lazy loading components, use React.lazy and Suspense directly in components.
// This file provides other performance utilities.

