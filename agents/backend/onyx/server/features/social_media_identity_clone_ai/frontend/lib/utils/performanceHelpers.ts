export const measurePerformance = (name: string, fn: () => void): void => {
  if (typeof window === 'undefined' || !('performance' in window)) {
    fn();
    return;
  }

  performance.mark(`${name}-start`);
  fn();
  performance.mark(`${name}-end`);
  performance.measure(name, `${name}-start`, `${name}-end`);

  const measure = performance.getEntriesByName(name)[0];
  console.log(`${name}: ${measure.duration.toFixed(2)}ms`);
};

export const measureAsyncPerformance = async <T,>(name: string, fn: () => Promise<T>): Promise<T> => {
  if (typeof window === 'undefined' || !('performance' in window)) {
    return fn();
  }

  performance.mark(`${name}-start`);
  const result = await fn();
  performance.mark(`${name}-end`);
  performance.measure(name, `${name}-start`, `${name}-end`);

  const measure = performance.getEntriesByName(name)[0];
  console.log(`${name}: ${measure.duration.toFixed(2)}ms`);

  return result;
};

export const getPerformanceMetrics = (): {
  navigation: PerformanceNavigationTiming | null;
  paint: PerformancePaintTiming[];
  resource: PerformanceResourceTiming[];
} => {
  if (typeof window === 'undefined' || !('performance' in window)) {
    return {
      navigation: null,
      paint: [],
      resource: [],
    };
  }

  const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
  const paint = performance.getEntriesByType('paint') as PerformancePaintTiming[];
  const resource = performance.getEntriesByType('resource') as PerformanceResourceTiming[];

  return {
    navigation: navigation || null,
    paint,
    resource,
  };
};

export const reportWebVitals = (onPerfEntry?: (metric: unknown) => void): void => {
  if (typeof window === 'undefined' || !onPerfEntry) {
    return;
  }

  import('web-vitals').then(({ onCLS, onFID, onFCP, onLCP, onTTFB }) => {
    onCLS(onPerfEntry);
    onFID(onPerfEntry);
    onFCP(onPerfEntry);
    onLCP(onPerfEntry);
    onTTFB(onPerfEntry);
  });
};



