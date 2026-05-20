import { useEffect, useRef } from 'react';
import { startTransaction, trackPerformance } from '@/utils/monitoring';

export function usePerformanceTracking(
  componentName: string,
  enabled = true
): void {
  const transactionRef = useRef<ReturnType<typeof startTransaction> | null>(
    null
  );

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const startTime = performance.now();
    transactionRef.current = startTransaction(`render.${componentName}`);

    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;

      trackPerformance({
        name: `component.render.${componentName}`,
        value: duration,
        unit: 'millisecond',
      });

      transactionRef.current?.finish();
    };
  }, [componentName, enabled]);
}

