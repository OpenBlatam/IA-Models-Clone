/**
 * Performance Monitoring Hook
 * React hook for monitoring component performance
 */

import { useEffect, useRef, useCallback } from 'react';
import { measurePerformance, mark, measure as measureTime } from '@/lib/performance/monitor';

interface UsePerformanceOptions {
  label?: string;
  enabled?: boolean;
  onMeasure?: (duration: number) => void;
}

/**
 * Hook to measure component render performance
 * @param options - Performance monitoring options
 * @returns Object with measure function and performance data
 */
export const usePerformance = (options: UsePerformanceOptions = {}) => {
  const { label, enabled = true, onMeasure } = options;
  const renderStartRef = useRef<number | null>(null);
  const renderCountRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;

    renderStartRef.current = performance.now();
    const markName = label ? `${label}-render-${renderCountRef.current}` : `render-${renderCountRef.current}`;
    mark(markName);

    return () => {
      if (renderStartRef.current) {
        const duration = performance.now() - renderStartRef.current;
        const measureName = `${markName}-complete`;
        measureTime(measureName, markName);
        onMeasure?.(duration);
        
        if (label && process.env.NODE_ENV === 'development') {
          console.log(`[Performance] ${label} render: ${duration.toFixed(2)}ms`);
        }
      }
      renderCountRef.current += 1;
    };
  });

  const measureFn = useCallback(
    async <T,>(fn: () => Promise<T> | T, fnLabel?: string): Promise<{ result: T; duration: number }> => {
      return measurePerformance(fn, fnLabel || label);
    },
    [label]
  );

  return {
    measure: measureFn,
    renderCount: renderCountRef.current,
  };
};

/**
 * Hook to measure async operation performance
 * @param label - Label for the operation
 * @returns Function to measure async operations
 */
export const useAsyncPerformance = (label?: string) => {
  return useCallback(
    async <T,>(fn: () => Promise<T>): Promise<{ result: T; duration: number }> => {
      return measurePerformance(fn, label);
    },
    [label]
  );
};


