/**
 * Hook for optimized animation frame handling
 * @module robot-3d-view/hooks/use-animation-frame
 */

import { useEffect, useRef } from 'react';

/**
 * Options for animation frame
 */
interface AnimationFrameOptions {
  /** Enable animation */
  enabled?: boolean;
  /** Target FPS (for throttling) */
  targetFPS?: number;
}

/**
 * Hook for optimized requestAnimationFrame
 * 
 * Provides a throttled animation frame callback for better performance.
 * 
 * @param callback - Callback function to run on each frame
 * @param options - Animation frame options
 * 
 * @example
 * ```tsx
 * useAnimationFrame((deltaTime) => {
 *   // Update animation
 * }, { targetFPS: 60 });
 * ```
 */
export function useAnimationFrame(
  callback: (deltaTime: number) => void,
  options: AnimationFrameOptions = {}
) {
  const { enabled = true, targetFPS = 60 } = options;
  const requestRef = useRef<number>();
  const previousTimeRef = useRef<number>();
  const frameInterval = 1000 / targetFPS;
  const lastFrameTimeRef = useRef<number>(0);

  useEffect(() => {
    if (!enabled) return;

    const animate = (currentTime: number) => {
      if (previousTimeRef.current === undefined) {
        previousTimeRef.current = currentTime;
      }

      const deltaTime = currentTime - previousTimeRef.current;
      const timeSinceLastFrame = currentTime - lastFrameTimeRef.current;

      // Throttle to target FPS
      if (timeSinceLastFrame >= frameInterval) {
        callback(deltaTime);
        lastFrameTimeRef.current = currentTime;
      }

      previousTimeRef.current = currentTime;
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [callback, enabled, frameInterval]);
}



