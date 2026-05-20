/**
 * Optimized @react-spring/web imports
 * 
 * This file provides optimized imports for react-spring to improve
 * tree-shaking and reduce bundle size.
 * 
 * @module robot-3d-view/lib/react-spring-imports
 */

// Main hooks and components
export {
  useSpring,
  useSprings,
  useTrail,
  useTransition,
  useChain,
  animated,
  config,
} from '@react-spring/web';

// Re-export types
export type {
  SpringValue,
  SpringRef,
  UseSpringProps,
  SpringConfig,
} from '@react-spring/web';

/**
 * Common spring configurations
 */
export const springConfigs = {
  gentle: { tension: 120, friction: 14 },
  wobbly: { tension: 180, friction: 12 },
  stiff: { tension: 210, friction: 20 },
  slow: { tension: 280, friction: 60 },
  default: { tension: 280, friction: 60 },
} as const;

/**
 * Helper to create spring with common config
 */
export function createSpring(
  from: Record<string, any>,
  to: Record<string, any>,
  configName: keyof typeof springConfigs = 'default'
) {
  return {
    from,
    to,
    config: springConfigs[configName],
  };
}



