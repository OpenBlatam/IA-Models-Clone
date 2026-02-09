/**
 * Advanced metrics and statistics system
 * @module robot-3d-view/utils/metrics
 */

/**
 * Metric type
 */
export type MetricType =
  | 'counter'
  | 'gauge'
  | 'histogram'
  | 'summary'
  | 'timer';

/**
 * Metric value
 */
export interface MetricValue {
  value: number;
  timestamp: number;
  labels?: Record<string, string>;
}

/**
 * Metric definition
 */
export interface Metric {
  name: string;
  type: MetricType;
  description?: string;
  unit?: string;
  values: MetricValue[];
  labels?: Record<string, string>;
}

/**
 * Metrics Manager class
 */
export class MetricsManager {
  private metrics: Map<string, Metric> = new Map();
  private maxValuesPerMetric = 1000;

  /**
   * Registers a metric
   */
  register(metric: Omit<Metric, 'values'>): void {
    this.metrics.set(metric.name, {
      ...metric,
      values: [],
    });
  }

  /**
   * Records a metric value
   */
  record(name: string, value: number, labels?: Record<string, string>): void {
    const metric = this.metrics.get(name);
    if (!metric) {
      console.warn(`Metric ${name} not registered`);
      return;
    }

    const metricValue: MetricValue = {
      value,
      timestamp: Date.now(),
      labels: labels || metric.labels,
    };

    metric.values.push(metricValue);

    // Limit values
    if (metric.values.length > this.maxValuesPerMetric) {
      metric.values.shift();
    }
  }

  /**
   * Increments a counter
   */
  increment(name: string, labels?: Record<string, string>): void {
    const metric = this.metrics.get(name);
    if (!metric) {
      this.register({
        name,
        type: 'counter',
      });
    }

    const lastValue = this.getLastValue(name) || 0;
    this.record(name, lastValue + 1, labels);
  }

  /**
   * Decrements a counter
   */
  decrement(name: string, labels?: Record<string, string>): void {
    const lastValue = this.getLastValue(name) || 0;
    this.record(name, Math.max(0, lastValue - 1), labels);
  }

  /**
   * Sets a gauge value
   */
  setGauge(name: string, value: number, labels?: Record<string, string>): void {
    const metric = this.metrics.get(name);
    if (!metric) {
      this.register({
        name,
        type: 'gauge',
      });
    }

    this.record(name, value, labels);
  }

  /**
   * Gets a metric
   */
  getMetric(name: string): Metric | undefined {
    return this.metrics.get(name);
  }

  /**
   * Gets all metrics
   */
  getAllMetrics(): Metric[] {
    return Array.from(this.metrics.values());
  }

  /**
   * Gets last value of a metric
   */
  getLastValue(name: string): number | undefined {
    const metric = this.metrics.get(name);
    if (!metric || metric.values.length === 0) return undefined;
    return metric.values[metric.values.length - 1].value;
  }

  /**
   * Gets average value
   */
  getAverage(name: string, window?: number): number | undefined {
    const metric = this.metrics.get(name);
    if (!metric || metric.values.length === 0) return undefined;

    const now = Date.now();
    const values = window
      ? metric.values.filter((v) => now - v.timestamp <= window)
      : metric.values;

    if (values.length === 0) return undefined;

    const sum = values.reduce((acc, v) => acc + v.value, 0);
    return sum / values.length;
  }

  /**
   * Gets min value
   */
  getMin(name: string, window?: number): number | undefined {
    const metric = this.metrics.get(name);
    if (!metric || metric.values.length === 0) return undefined;

    const now = Date.now();
    const values = window
      ? metric.values.filter((v) => now - v.timestamp <= window)
      : metric.values;

    if (values.length === 0) return undefined;

    return Math.min(...values.map((v) => v.value));
  }

  /**
   * Gets max value
   */
  getMax(name: string, window?: number): number | undefined {
    const metric = this.metrics.get(name);
    if (!metric || metric.values.length === 0) return undefined;

    const now = Date.now();
    const values = window
      ? metric.values.filter((v) => now - v.timestamp <= window)
      : metric.values;

    if (values.length === 0) return undefined;

    return Math.max(...values.map((v) => v.value));
  }

  /**
   * Gets percentile
   */
  getPercentile(name: string, percentile: number, window?: number): number | undefined {
    const metric = this.metrics.get(name);
    if (!metric || metric.values.length === 0) return undefined;

    const now = Date.now();
    const values = window
      ? metric.values.filter((v) => now - v.timestamp <= window)
      : metric.values;

    if (values.length === 0) return undefined;

    const sorted = [...values.map((v) => v.value)].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Resets a metric
   */
  reset(name: string): void {
    const metric = this.metrics.get(name);
    if (metric) {
      metric.values = [];
    }
  }

  /**
   * Clears all metrics
   */
  clear(): void {
    this.metrics.clear();
  }

  /**
   * Exports metrics
   */
  export(): string {
    return JSON.stringify(
      Array.from(this.metrics.values()),
      null,
      2
    );
  }
}

/**
 * Global metrics manager instance
 */
export const metricsManager = new MetricsManager();

// Register default metrics
metricsManager.register({
  name: 'render.fps',
  type: 'gauge',
  description: 'Frames per second',
  unit: 'fps',
});

metricsManager.register({
  name: 'render.frameTime',
  type: 'histogram',
  description: 'Frame render time',
  unit: 'ms',
});

metricsManager.register({
  name: 'interaction.clicks',
  type: 'counter',
  description: 'Total clicks',
});

metricsManager.register({
  name: 'interaction.shortcuts',
  type: 'counter',
  description: 'Total shortcuts used',
});

metricsManager.register({
  name: 'config.changes',
  type: 'counter',
  description: 'Total configuration changes',
});



