'use client';

import { useEffect, useMemo } from 'react';
import { useAppStore } from '@/store/app-store';

interface PerformanceMetrics {
  renderTime: number;
  componentCount: number;
  memoryUsage?: number;
}

export function usePerformanceOptimizer() {
  const { activeView } = useAppStore();

  useEffect(() => {
    // Lazy load components based on view
    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      requestIdleCallback(() => {
        // Preload next likely view
        const preloadMap: Record<string, string[]> = {
          dashboard: ['tasks', 'generate'],
          generate: ['tasks', 'documents'],
          tasks: ['documents', 'stats'],
        };

        const toPreload = preloadMap[activeView] || [];
        toPreload.forEach((view) => {
          // Prefetch route data
          if (typeof window !== 'undefined') {
            const link = document.createElement('link');
            link.rel = 'prefetch';
            link.href = `/${view}`;
            document.head.appendChild(link);
          }
        });
      });
    }
  }, [activeView]);
}

export function useMemoizedValue<T>(value: T, deps: any[]): T {
  return useMemo(() => value, deps);
}

export default function PerformanceOptimizer() {
  usePerformanceOptimizer();
  return null;
}

