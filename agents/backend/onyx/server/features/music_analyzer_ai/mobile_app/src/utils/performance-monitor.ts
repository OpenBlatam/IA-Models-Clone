/**
 * Performance monitoring utilities
 * Tracks app performance metrics
 */

interface PerformanceMetric {
  name: string;
  duration: number;
  timestamp: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private marks: Map<string, number> = new Map();

  /**
   * Mark the start of a performance measurement
   */
  mark(name: string): void {
    this.marks.set(name, performance.now());
  }

  /**
   * Measure the duration since a mark
   */
  measure(name: string, markName?: string): number {
    const endTime = performance.now();
    const startTime = markName
      ? this.marks.get(markName) ?? endTime
      : this.marks.get(name) ?? endTime;

    const duration = endTime - startTime;

    this.metrics.push({
      name,
      duration,
      timestamp: Date.now(),
    });

    return duration;
  }

  /**
   * Get all metrics
   */
  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  /**
   * Get metrics by name
   */
  getMetricsByName(name: string): PerformanceMetric[] {
    return this.metrics.filter((metric) => metric.name === name);
  }

  /**
   * Get average duration for a metric
   */
  getAverageDuration(name: string): number {
    const metrics = this.getMetricsByName(name);
    if (metrics.length === 0) {
      return 0;
    }

    const sum = metrics.reduce((acc, metric) => acc + metric.duration, 0);
    return sum / metrics.length;
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics = [];
    this.marks.clear();
  }

  /**
   * Log performance report
   */
  logReport(): void {
    const report = this.metrics
      .map(
        (metric) =>
          `${metric.name}: ${metric.duration.toFixed(2)}ms (${new Date(metric.timestamp).toISOString()})`
      )
      .join('\n');

    console.log('Performance Report:\n', report);
  }
}

export const performanceMonitor = new PerformanceMonitor();

/**
 * Measure function execution time
 */
export function measurePerformance<T>(
  name: string,
  fn: () => T
): T {
  performanceMonitor.mark(name);
  const result = fn();
  const duration = performanceMonitor.measure(name);
  
  if (__DEV__) {
    console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
  }
  
  return result;
}

/**
 * Measure async function execution time
 */
export async function measureAsyncPerformance<T>(
  name: string,
  fn: () => Promise<T>
): Promise<T> {
  performanceMonitor.mark(name);
  const result = await fn();
  const duration = performanceMonitor.measure(name);
  
  if (__DEV__) {
    console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
  }
  
  return result;
}

