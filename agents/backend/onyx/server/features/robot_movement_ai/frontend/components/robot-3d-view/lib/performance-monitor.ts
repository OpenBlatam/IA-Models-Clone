/**
 * Performance monitoring utilities
 * @module robot-3d-view/lib/performance-monitor
 */

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  renderTime: number;
  memoryUsage?: number;
  timestamp: number;
}

/**
 * Performance monitor class
 */
export class PerformanceMonitor {
  private frameCount = 0;
  private lastTime = performance.now();
  private fps = 60;
  private frameTime = 16.67;
  private renderTime = 0;
  private samples: PerformanceMetrics[] = [];
  private readonly maxSamples = 100;
  private readonly lowFPSThreshold = 30;
  private callbacks: Set<(metrics: PerformanceMetrics) => void> = new Set();

  /**
   * Records a frame
   */
  recordFrame(renderTime?: number): void {
    const currentTime = performance.now();
    const delta = currentTime - this.lastTime;

    this.frameCount++;
    this.frameTime = delta;

    if (renderTime !== undefined) {
      this.renderTime = renderTime;
    }

    // Calculate FPS every second
    if (delta >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / delta);
      this.frameCount = 0;
      this.lastTime = currentTime;

      const metrics: PerformanceMetrics = {
        fps: this.fps,
        frameTime: this.frameTime,
        renderTime: this.renderTime,
        memoryUsage: this.getMemoryUsage(),
        timestamp: currentTime,
      };

      this.samples.push(metrics);
      if (this.samples.length > this.maxSamples) {
        this.samples.shift();
      }

      // Notify callbacks
      this.callbacks.forEach((callback) => callback(metrics));

      // Warn if FPS is low
      if (this.fps < this.lowFPSThreshold) {
        console.warn(`Low FPS detected: ${this.fps} fps`);
      }
    }
  }

  /**
   * Gets current metrics
   */
  getMetrics(): PerformanceMetrics {
    return {
      fps: this.fps,
      frameTime: this.frameTime,
      renderTime: this.renderTime,
      memoryUsage: this.getMemoryUsage(),
      timestamp: performance.now(),
    };
  }

  /**
   * Gets average metrics over time
   */
  getAverageMetrics(): Partial<PerformanceMetrics> {
    if (this.samples.length === 0) {
      return {};
    }

    const sum = this.samples.reduce(
      (acc, sample) => ({
        fps: acc.fps + sample.fps,
        frameTime: acc.frameTime + sample.frameTime,
        renderTime: acc.renderTime + sample.renderTime,
      }),
      { fps: 0, frameTime: 0, renderTime: 0 }
    );

    const count = this.samples.length;
    return {
      fps: Math.round(sum.fps / count),
      frameTime: sum.frameTime / count,
      renderTime: sum.renderTime / count,
    };
  }

  /**
   * Subscribes to performance updates
   */
  subscribe(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.callbacks.add(callback);
    return () => {
      this.callbacks.delete(callback);
    };
  }

  /**
   * Gets memory usage (if available)
   */
  private getMemoryUsage(): number | undefined {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return memory.usedJSHeapSize / 1048576; // Convert to MB
    }
    return undefined;
  }

  /**
   * Resets the monitor
   */
  reset(): void {
    this.frameCount = 0;
    this.lastTime = performance.now();
    this.samples = [];
    this.fps = 60;
    this.frameTime = 16.67;
    this.renderTime = 0;
  }
}

/**
 * Global performance monitor instance
 */
export const performanceMonitor = new PerformanceMonitor();



