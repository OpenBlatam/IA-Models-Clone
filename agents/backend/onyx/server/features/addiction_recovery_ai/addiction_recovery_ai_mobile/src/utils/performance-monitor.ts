interface PerformanceMetric {
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
}

class PerformanceMonitor {
  private metrics = new Map<string, PerformanceMetric>();
  private enabled = __DEV__;

  start(name: string): void {
    if (!this.enabled) {
      return;
    }

    this.metrics.set(name, {
      name,
      startTime: performance.now(),
    });
  }

  end(name: string): number | null {
    if (!this.enabled) {
      return null;
    }

    const metric = this.metrics.get(name);
    if (!metric) {
      return null;
    }

    const endTime = performance.now();
    const duration = endTime - metric.startTime;

    metric.endTime = endTime;
    metric.duration = duration;

    if (duration > 16) {
      console.warn(`[Performance] ${name} took ${duration.toFixed(2)}ms`);
    }

    return duration;
  }

  measure<T>(name: string, fn: () => T): T {
    this.start(name);
    try {
      const result = fn();
      this.end(name);
      return result;
    } catch (error) {
      this.end(name);
      throw error;
    }
  }

  async measureAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    this.start(name);
    try {
      const result = await fn();
      this.end(name);
      return result;
    } catch (error) {
      this.end(name);
      throw error;
    }
  }

  getMetrics(): PerformanceMetric[] {
    return Array.from(this.metrics.values());
  }

  clear(): void {
    this.metrics.clear();
  }
}

export const performanceMonitor = new PerformanceMonitor();

