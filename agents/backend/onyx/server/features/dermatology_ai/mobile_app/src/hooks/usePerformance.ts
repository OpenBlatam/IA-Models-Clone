import { useEffect, useRef } from 'react';

interface PerformanceMetrics {
  renderTime: number;
  componentName: string;
}

export const usePerformance = (componentName: string) => {
  const renderStartTime = useRef<number>(Date.now());

  useEffect(() => {
    const renderTime = Date.now() - renderStartTime.current;
    
    if (__DEV__) {
      console.log(`[Performance] ${componentName} rendered in ${renderTime}ms`);
    }

    // Track performance metrics
    if (renderTime > 100) {
      console.warn(`[Performance Warning] ${componentName} took ${renderTime}ms to render`);
    }
    
    renderStartTime.current = Date.now();
  });

  const measureAsync = async <T,>(
    operationName: string,
    operation: () => Promise<T>
  ): Promise<T> => {
    const startTime = Date.now();
    try {
      const result = await operation();
      const duration = Date.now() - startTime;
      
      if (__DEV__) {
        console.log(`[Performance] ${operationName} took ${duration}ms`);
      }
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(`[Performance] ${operationName} failed after ${duration}ms:`, error);
      throw error;
    }
  };

  return { measureAsync };
};
