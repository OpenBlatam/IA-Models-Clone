/**
 * Custom hook for performance monitoring.
 * Provides reactive performance monitoring functionality.
 */

import { useRef, useCallback, useEffect } from 'react';
import {
  PerformanceMonitor,
  measurePerformance,
  measurePerformanceAsync,
} from '../utils/performance-monitor';

/**
 * Custom hook for performance monitoring.
 * Provides reactive performance monitoring functionality.
 *
 * @returns Performance monitoring operations
 */
export function usePerformanceMonitor() {
  const monitorRef = useRef<PerformanceMonitor>(new PerformanceMonitor());

  const mark = useCallback((name: string) => {
    monitorRef.current.mark(name);
  }, []);

  const measure = useCallback(
    (name: string, metadata?: Record<string, any>) => {
      return monitorRef.current.measure(name, metadata);
    },
    []
  );

  const getMetrics = useCallback(() => {
    return monitorRef.current.getMetrics();
  }, []);

  const getMetricsByName = useCallback((name: string) => {
    return monitorRef.current.getMetricsByName(name);
  }, []);

  const getAverageDuration = useCallback((name: string) => {
    return monitorRef.current.getAverageDuration(name);
  }, []);

  const clear = useCallback(() => {
    monitorRef.current.clear();
  }, []);

  const exportMetrics = useCallback(() => {
    return monitorRef.current.export();
  }, []);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      monitorRef.current.clear();
    };
  }, []);

  return {
    mark,
    measure,
    getMetrics,
    getMetricsByName,
    getAverageDuration,
    clear,
    exportMetrics,
    measurePerformance: <T,>(name: string, fn: () => T, metadata?: Record<string, any>) => {
      return measurePerformance(name, fn, metadata);
    },
    measurePerformanceAsync: <T,>(
      name: string,
      fn: () => Promise<T>,
      metadata?: Record<string, any>
    ) => {
      return measurePerformanceAsync(name, fn, metadata);
    },
  };
}

