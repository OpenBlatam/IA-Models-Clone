/**
 * Type definitions for Robot 3D View components
 * 
 * @deprecated Use schemas/validation-schemas.ts for type-safe Zod schemas
 * @module robot-3d-view/types
 * 
 * This file is kept for backward compatibility.
 * New code should import types from schemas/validation-schemas.ts
 */

// Re-export types from validation schemas for backward compatibility
export type {
  Position3D,
  CameraPreset,
  RobotStatus,
  ViewConfig,
  SceneConfig,
  Waypoint,
  MaterialProps,
  AnimationConfig,
  BaseObjectProps,
  Robot3DViewProps,
} from './schemas/validation-schemas';

/**
 * Obstacle bounds [minX, minY, minZ, maxX, maxY, maxZ]
 */
export type ObstacleBounds = [number, number, number, number, number, number];

/**
 * Robot arm component props
 */
export interface RobotArmProps {
  position: Position3D;
  hovered?: boolean;
}

/**
 * Target marker props
 */
export interface TargetMarkerProps {
  position: Position3D;
  label?: string;
}

/**
 * Trajectory path props
 */
export interface TrajectoryPathProps {
  waypoints: Waypoint[];
  showWaypoints?: boolean;
}

/**
 * Sensor component props
 */
export interface SensorProps {
  position: Position3D;
  rotation?: Position3D;
  intensity?: number;
  status?: RobotStatus;
}

