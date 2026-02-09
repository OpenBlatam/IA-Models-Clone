/**
 * Robot 3D View - Main Component
 * 
 * Optimized and modular 3D visualization component for robot movement.
 * Uses code splitting, memoization, and best practices for performance.
 * 
 * This is a Server Component wrapper that delegates to client components.
 * 
 * @module robot-3d-view
 * @example
 * ```tsx
 * <Robot3DView fullscreen={false} />
 * ```
 */

import { Robot3DViewContainer } from './components/robot-3d-view-container';
import type { Robot3DViewProps } from './schemas/validation-schemas';

/**
 * Main Robot 3D View Component
 * 
 * Server Component that provides a comprehensive 3D visualization of the robot.
 * 
 * Features:
 * - Real-time position tracking
 * - Trajectory visualization
 * - Interactive controls
 * - Performance monitoring
 * - Customizable view options
 * 
 * @param props - Component props
 * @returns Robot 3D view component
 */
export default function Robot3DView(props: Robot3DViewProps) {
  return <Robot3DViewContainer {...props} />;
}

// Re-export types for convenience
export type { Robot3DViewProps } from './schemas/validation-schemas';

