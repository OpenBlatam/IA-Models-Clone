/**
 * Performance monitoring utility functions.
 * Provides helper functions for performance monitoring and metrics.
 */

/**
 * Performance metric.
 */
export interface PerformanceMetric {
  name: string;
  duration: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

/**
 * Performance monitor class.
 */
export class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private marks: Map<string, number> = new Map();

  /**
   * Marks the start of a performance measurement.
   */
  mark(name: string): void {
    if (typeof performance !== 'undefined' && performance.mark) {
      performance.mark(`${name}-start`);
    }
    this.marks.set(name, performance.now());
  }

  /**
   * Measures the duration since the last mark.
   */
  measure(name: string, metadata?: Record<string, any>): number {
    const startTime = this.marks.get(name);
    if (!startTime) {
      console.warn(`No mark found for: ${name}`);
      return 0;
    }

    const duration = performance.now() - startTime;
    const metric: PerformanceMetric = {
      name,
      duration,
      timestamp: Date.now(),
      metadata,
    };

    this.metrics.push(metric);

    if (typeof performance !== 'undefined' && performance.mark) {
      performance.mark(`${name}-end`);
      performance.measure(name, `${name}-start`, `${name}-end`);
    }

    this.marks.delete(name);
    return duration;
  }

  /**
   * Gets all metrics.
   */
  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  /**
   * Gets metrics by name.
   */
  getMetricsByName(name: string): PerformanceMetric[] {
    return this.metrics.filter((m) => m.name === name);
  }

  /**
   * Gets average duration for a metric name.
   */
  getAverageDuration(name: string): number {
    const metrics = this.getMetricsByName(name);
    if (metrics.length === 0) return 0;
    const sum = metrics.reduce((acc, m) => acc + m.duration, 0);
    return sum / metrics.length;
  }

  /**
   * Clears all metrics.
   */
  clear(): void {
    this.metrics = [];
    this.marks.clear();
  }

  /**
   * Exports metrics as JSON.
   */
  export(): string {
    return JSON.stringify(this.metrics, null, 2);
  }
}

/**
 * Global performance monitor instance.
 */
export const performanceMonitor = new PerformanceMonitor();

/**
 * Measures function execution time.
 */
export function measurePerformance<T>(
  name: string,
  fn: () => T,
  metadata?: Record<string, any>
): T {
  performanceMonitor.mark(name);
  try {
    const result = fn();
    performanceMonitor.measure(name, metadata);
    return result;
  } catch (error) {
    performanceMonitor.measure(name, { ...metadata, error: true });
    throw error;
  }
}

/**
 * Measures async function execution time.
 */
export async function measurePerformanceAsync<T>(
  name: string,
  fn: () => Promise<T>,
  metadata?: Record<string, any>
): Promise<T> {
  performanceMonitor.mark(name);
  try {
    const result = await fn();
    performanceMonitor.measure(name, metadata);
    return result;
  } catch (error) {
    performanceMonitor.measure(name, { ...metadata, error: true });
    throw error;
  }
}

