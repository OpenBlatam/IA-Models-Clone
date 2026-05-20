import { useEffect, useRef, useState, useCallback } from 'react';

interface PerformanceMetrics {
  fps: number;
  renderTime: number;
  memoryUsage?: number;
}

interface UsePerformanceOptions {
  enabled?: boolean;
  sampleSize?: number;
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void;
}

export function usePerformance(options: UsePerformanceOptions = {}) {
  const { enabled = __DEV__, sampleSize = 60, onMetricsUpdate } = options;
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    renderTime: 0,
  });

  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const renderStartRef = useRef(performance.now());
  const samplesRef = useRef<number[]>([]);
  const animationFrameRef = useRef<number>();

  const measureRender = useCallback(() => {
    const renderTime = performance.now() - renderStartRef.current;
    samplesRef.current.push(renderTime);

    if (samplesRef.current.length > sampleSize) {
      samplesRef.current.shift();
    }

    const averageRenderTime =
      samplesRef.current.reduce((a, b) => a + b, 0) / samplesRef.current.length;

    return averageRenderTime;
  }, [sampleSize]);

  useEffect(() => {
    if (!enabled) return;

    const measureFPS = () => {
      frameCountRef.current += 1;
      const now = performance.now();
      const elapsed = now - lastTimeRef.current;

      if (elapsed >= 1000) {
        const currentFPS = Math.round((frameCountRef.current * 1000) / elapsed);
        const renderTime = measureRender();

        const newMetrics: PerformanceMetrics = {
          fps: currentFPS,
          renderTime: Math.round(renderTime * 100) / 100,
        };

        if ('performance' in global && 'memory' in global.performance) {
          const perf = global.performance as Performance & {
            memory?: { usedJSHeapSize: number };
          };
          if (perf.memory) {
            newMetrics.memoryUsage = Math.round(
              perf.memory.usedJSHeapSize / 1048576
            );
          }
        }

        setMetrics(newMetrics);
        onMetricsUpdate?.(newMetrics);

        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }

      animationFrameRef.current = requestAnimationFrame(measureFPS);
    };

    renderStartRef.current = performance.now();
    animationFrameRef.current = requestAnimationFrame(measureFPS);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, measureRender, onMetricsUpdate]);

  const startRenderMeasurement = useCallback(() => {
    renderStartRef.current = performance.now();
  }, []);

  return {
    metrics,
    startRenderMeasurement,
  };
}
