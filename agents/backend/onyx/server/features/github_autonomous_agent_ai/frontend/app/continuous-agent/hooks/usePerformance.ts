/**
 * Custom hook for performance monitoring
 * 
 * Provides React integration for performance measurements
 */

import { useCallback, useRef } from "react";
import { PerformanceMonitor } from "../utils/monitoring/logger";

/**
 * Return type for usePerformance hook
 */
export interface UsePerformanceReturn {
  /** Start measurement */
  readonly start: (label: string) => void;
  /** End measurement and get duration */
  readonly end: (label: string) => number;
  /** Measure async function */
  readonly measure: <T>(label: string, fn: () => Promise<T>) => Promise<T>;
}

/**
 * Custom hook for performance monitoring
 * 
 * @returns Performance measurement functions
 * 
 * @example
 * ```typescript
 * const { start, end, measure } = usePerformance();
 * 
 * start("fetchAgents");
 * const agents = await fetchAgents();
 * const duration = end("fetchAgents");
 * 
 * // Or use measure helper
 * const agents = await measure("fetchAgents", () => fetchAgents());
 * ```
 */
export function usePerformance(): UsePerformanceReturn {
  const monitorRef = useRef<PerformanceMonitor>(new PerformanceMonitor());

  const start = useCallback((label: string) => {
    monitorRef.current.start(label);
  }, []);

  const end = useCallback((label: string) => {
    return monitorRef.current.end(label);
  }, []);

  const measure = useCallback(
    async <T>(label: string, fn: () => Promise<T>): Promise<T> => {
      return monitorRef.current.measure(label, fn);
    },
    []
  );

  return {
    start,
    end,
    measure,
  };
}




