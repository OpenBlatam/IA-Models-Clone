/**
 * Hook for spring physics animations
 * @module robot-3d-view/hooks/use-spring-physics
 */

import { useState, useEffect, useRef } from 'react';

/**
 * Spring physics configuration
 */
interface SpringConfig {
  /** Spring stiffness */
  stiffness?: number;
  /** Damping coefficient */
  damping?: number;
  /** Mass */
  mass?: number;
  /** Initial velocity */
  initialVelocity?: number;
}

/**
 * Spring state
 */
interface SpringState {
  value: number;
  velocity: number;
}

const defaultConfig: Required<SpringConfig> = {
  stiffness: 100,
  damping: 10,
  mass: 1,
  initialVelocity: 0,
};

/**
 * Hook for spring physics animation
 * 
 * Provides smooth spring-based animations for numeric values.
 * 
 * @param target - Target value
 * @param config - Spring configuration
 * @returns Current spring value
 * 
 * @example
 * ```tsx
 * const springValue = useSpringPhysics(targetValue, {
 *   stiffness: 100,
 *   damping: 10,
 * });
 * ```
 */
export function useSpringPhysics(target: number, config: SpringConfig = {}): number {
  const fullConfig = { ...defaultConfig, ...config };
  const [state, setState] = useState<SpringState>({
    value: target,
    velocity: fullConfig.initialVelocity,
  });
  const animationFrameRef = useRef<number>();
  const lastTimeRef = useRef<number>(performance.now());

  useEffect(() => {
    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTimeRef.current) / 1000; // Convert to seconds
      lastTimeRef.current = currentTime;

      setState((prevState) => {
        const displacement = target - prevState.value;
        const springForce = fullConfig.stiffness * displacement;
        const dampingForce = fullConfig.damping * prevState.velocity;
        const acceleration = (springForce - dampingForce) / fullConfig.mass;

        const newVelocity = prevState.velocity + acceleration * deltaTime;
        const newValue = prevState.value + newVelocity * deltaTime;

        // Stop animation if very close to target
        if (Math.abs(displacement) < 0.001 && Math.abs(newVelocity) < 0.001) {
          return { value: target, velocity: 0 };
        }

        return { value: newValue, velocity: newVelocity };
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [target, fullConfig.stiffness, fullConfig.damping, fullConfig.mass]);

  return state.value;
}



