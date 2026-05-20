import { useEffect, useRef } from 'react';
import { performanceMonitor } from '@/utils/performance-monitor';

export function usePerformanceMeasure(
  componentName: string,
  enabled = __DEV__
): void {
  const measureRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    measureRef.current = `render.${componentName}`;
    performanceMonitor.start(measureRef.current);

    return () => {
      if (measureRef.current) {
        performanceMonitor.end(measureRef.current);
      }
    };
  }, [componentName, enabled]);
}

