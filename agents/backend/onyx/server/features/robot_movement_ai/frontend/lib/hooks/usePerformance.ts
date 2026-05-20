import { useEffect, useState } from 'react';
import { getPerformanceMetrics } from '@/lib/utils/performance';

export function usePerformance() {
  const [metrics, setMetrics] = useState<ReturnType<typeof getPerformanceMetrics>>(null);

  useEffect(() => {
    // Get initial metrics
    const initialMetrics = getPerformanceMetrics();
    setMetrics(initialMetrics);

    // Monitor performance
    if (typeof window !== 'undefined' && 'PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'measure') {
            console.log(`[Performance] ${entry.name}: ${entry.duration.toFixed(2)}ms`);
          }
        });
      });

      try {
        observer.observe({ entryTypes: ['measure', 'navigation'] });
      } catch (e) {
        console.warn('PerformanceObserver not supported');
      }

      return () => {
        observer.disconnect();
      };
    }
  }, []);

  return metrics;
}



