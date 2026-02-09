/**
 * Performance monitoring hook
 * @module robot-3d-view/hooks/use-performance
 */

import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';

/**
 * Options for performance monitoring
 */
interface PerformanceOptions {
  /** Enable FPS monitoring */
  monitorFPS?: boolean;
  /** FPS threshold for warnings */
  fpsThreshold?: number;
  /** Callback when FPS drops below threshold */
  onLowFPS?: (fps: number) => void;
}

/**
 * Hook for monitoring 3D scene performance
 * 
 * @param options - Performance monitoring options
 * @returns Performance metrics
 * 
 * @example
 * ```tsx
 * const { fps } = usePerformance({ monitorFPS: true, fpsThreshold: 30 });
 * ```
 */
export function usePerformance(options: PerformanceOptions = {}) {
  const { monitorFPS = false, fpsThreshold = 30, onLowFPS } = options;
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const fpsRef = useRef(60);

  useFrame(() => {
    if (!monitorFPS) return;

    frameCount.current++;
    const currentTime = performance.now();
    const delta = currentTime - lastTime.current;

    if (delta >= 1000) {
      const fps = Math.round((frameCount.current * 1000) / delta);
      fpsRef.current = fps;

      if (fps < fpsThreshold && onLowFPS) {
        onLowFPS(fps);
      }

      frameCount.current = 0;
      lastTime.current = currentTime;
    }
  });

  return {
    fps: fpsRef.current,
  };
}



