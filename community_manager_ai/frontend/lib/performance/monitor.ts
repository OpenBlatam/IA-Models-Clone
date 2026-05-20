/**
 * Performance Monitoring Utilities
 * Provides utilities for monitoring and optimizing application performance
 */

/**
 * Measures the execution time of a function
 * @param fn - Function to measure
 * @param label - Optional label for logging
 * @returns The result of the function and execution time
 */
export const measurePerformance = async <T>(
  fn: () => Promise<T> | T,
  label?: string
): Promise<{ result: T; duration: number }> => {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;

  if (label && process.env.NODE_ENV === 'development') {
    console.log(`[Performance] ${label}: ${duration.toFixed(2)}ms`);
  }

  return { result, duration };
};

/**
 * Creates a performance mark
 * @param name - Mark name
 */
export const mark = (name: string): void => {
  if (typeof performance !== 'undefined' && performance.mark) {
    performance.mark(name);
  }
};

/**
 * Measures the time between two marks
 * @param markName - Name of the mark to measure
 * @param startMark - Name of the start mark (optional)
 * @returns Duration in milliseconds
 */
export const measure = (markName: string, startMark?: string): number | null => {
  if (typeof performance === 'undefined' || !performance.measure) {
    return null;
  }

  try {
    const measureName = startMark ? `${startMark}-to-${markName}` : markName;
    performance.measure(measureName, startMark, markName);
    const measure = performance.getEntriesByName(measureName)[0];
    return measure?.duration ?? null;
  } catch {
    return null;
  }
};

/**
 * Debounces a function with performance tracking
 * @param fn - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export const debounceWithPerformance = <T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number,
  label?: string
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      if (label) {
        measurePerformance(() => fn(...args), label);
      } else {
        fn(...args);
      }
    }, delay);
  };
};

/**
 * Throttles a function with performance tracking
 * @param fn - Function to throttle
 * @param limit - Time limit in milliseconds
 * @returns Throttled function
 */
export const throttleWithPerformance = <T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number,
  label?: string
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      if (label) {
        measurePerformance(() => fn(...args), label);
      } else {
        fn(...args);
      }
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
};

/**
 * Gets Web Vitals metrics
 * @returns Promise with Web Vitals data
 */
export const getWebVitals = async (): Promise<{
  fcp?: number;
  lcp?: number;
  fid?: number;
  cls?: number;
  ttfb?: number;
}> => {
  if (typeof window === 'undefined') {
    return {};
  }

  return new Promise((resolve) => {
    const vitals: Record<string, number> = {};

    // First Contentful Paint
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.name === 'first-contentful-paint') {
              vitals.fcp = entry.startTime;
            }
          }
        });
        observer.observe({ entryTypes: ['paint'] });
      } catch {
        // PerformanceObserver not supported
      }
    }

    // Largest Contentful Paint
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1] as PerformanceEntry & {
            renderTime?: number;
            loadTime?: number;
          };
          if (lastEntry) {
            vitals.lcp = lastEntry.renderTime || lastEntry.loadTime || 0;
          }
        });
        observer.observe({ entryTypes: ['largest-contentful-paint'] });
      } catch {
        // PerformanceObserver not supported
      }
    }

    // Time to First Byte
    if (performance.timing) {
      vitals.ttfb = performance.timing.responseStart - performance.timing.requestStart;
    }

    // Resolve after a short delay to allow metrics to be collected
    setTimeout(() => resolve(vitals), 2000);
  });
};


