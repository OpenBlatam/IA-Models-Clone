"use client";

import { useEffect } from "react";
import { usePerformance } from "@/components/providers/performance-provider";

export function PerformanceMonitor() {
  const { isSlowConnection } = usePerformance();

  useEffect(() => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            console.log('Navigation timing:', {
              domContentLoaded: navEntry.domContentLoadedEventEnd - navEntry.domContentLoadedEventStart,
              loadComplete: navEntry.loadEventEnd - navEntry.loadEventStart,
              firstPaint: navEntry.responseEnd - navEntry.requestStart
            });
          }
        }
      });

      observer.observe({ entryTypes: ['navigation', 'paint', 'largest-contentful-paint'] });

      return () => observer.disconnect();
    }
  }, []);

  useEffect(() => {
    if (isSlowConnection) {
      console.log('Slow connection detected - optimizing performance');
    }
  }, [isSlowConnection]);

  useEffect(() => {
    const handleRouteChange = () => {
      const now = performance.now();
      console.log('Route change at:', now);
    };

    window.addEventListener('beforeunload', handleRouteChange);
    return () => window.removeEventListener('beforeunload', handleRouteChange);
  }, []);

  return null;
}
