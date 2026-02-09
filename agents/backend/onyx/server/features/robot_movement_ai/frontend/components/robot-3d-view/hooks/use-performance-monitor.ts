/**
 * Hook for performance monitoring
 * @module robot-3d-view/hooks/use-performance-monitor
 */

import { useEffect, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { performanceMonitor, type PerformanceMetrics } from '../lib/performance-monitor';

/**
 * Options for performance monitoring
 */
interface UsePerformanceMonitorOptions {
  /** Enable monitoring */
  enabled?: boolean;
  /** Callback when metrics update */
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void;
  /** Low FPS threshold */
  lowFPSThreshold?: number;
}

/**
 * Hook for monitoring 3D scene performance
 * 
 * @param options - Monitoring options
 * @returns Performance metrics and controls
 * 
 * @example
 * ```tsx
 * const { metrics, isLowPerformance } = usePerformanceMonitor({
 *   enabled: true,
 *   onMetricsUpdate: (m) => console.log('FPS:', m.fps),
 * });
 * ```
 */
export function usePerformanceMonitor(
  options: UsePerformanceMonitorOptions = {}
) {
  const {
    enabled = false,
    onMetricsUpdate,
    lowFPSThreshold = 30,
  } = options;

  const [metrics, setMetrics] = useState<PerformanceMetrics>(
    performanceMonitor.getMetrics()
  );
  const renderStartTime = useRef<number>(0);

  // Monitor frame performance
  useFrame(() => {
    if (!enabled) return;

    const renderTime = performance.now() - renderStartTime.current;
    performanceMonitor.recordFrame(renderTime);
    renderStartTime.current = performance.now();
  });

  // Subscribe to metrics updates
  useEffect(() => {
    if (!enabled) return;

    const unsubscribe = performanceMonitor.subscribe((newMetrics) => {
      setMetrics(newMetrics);
      onMetricsUpdate?.(newMetrics);
    });

    return unsubscribe;
  }, [enabled, onMetricsUpdate]);

  const isLowPerformance = metrics.fps < lowFPSThreshold;
  const averageMetrics = enabled
    ? performanceMonitor.getAverageMetrics()
    : undefined;

  return {
    metrics,
    isLowPerformance,
    averageMetrics,
    reset: () => performanceMonitor.reset(),
  };
}



