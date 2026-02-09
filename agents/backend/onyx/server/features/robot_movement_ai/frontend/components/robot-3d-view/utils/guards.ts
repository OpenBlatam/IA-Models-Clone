/**
 * Guard functions for type checking and validation
 * @module robot-3d-view/utils/guards
 */

import type { Position3D, Waypoint, SceneConfig } from '../types';

/**
 * Type guard for Position3D
 * 
 * @param value - Value to check
 * @returns true if value is a valid Position3D
 */
export function isPosition3D(value: unknown): value is Position3D {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((coord) => typeof coord === 'number' && !isNaN(coord))
  );
}

/**
 * Type guard for Waypoint array
 * 
 * @param value - Value to check
 * @returns true if value is a valid Waypoint array
 */
export function isWaypointArray(value: unknown): value is Waypoint[] {
  return Array.isArray(value) && value.every(isPosition3D);
}

/**
 * Type guard for SceneConfig
 * 
 * @param value - Value to check
 * @returns true if value is a valid SceneConfig
 */
export function isSceneConfig(value: unknown): value is SceneConfig {
  if (typeof value !== 'object' || value === null) return false;

  const config = value as Record<string, unknown>;

  return (
    typeof config.showStats === 'boolean' &&
    typeof config.showGizmo === 'boolean' &&
    typeof config.showStars === 'boolean' &&
    typeof config.showWaypoints === 'boolean' &&
    typeof config.showGrid === 'boolean' &&
    typeof config.showObjects === 'boolean' &&
    typeof config.autoRotate === 'boolean' &&
    typeof config.gridSize === 'number' &&
    (config.cameraPreset === null || typeof config.cameraPreset === 'string')
  );
}

/**
 * Safe position converter with validation
 * 
 * @param position - Position to convert
 * @param fallback - Fallback position if invalid
 * @returns Valid Position3D
 */
export function safePosition3D(
  position: unknown,
  fallback: Position3D = [0, 0, 0]
): Position3D {
  if (isPosition3D(position)) {
    return position;
  }

  if (
    typeof position === 'object' &&
    position !== null &&
    'x' in position &&
    'y' in position &&
    'z' in position
  ) {
    const pos = position as { x: unknown; y: unknown; z: unknown };
    const x = typeof pos.x === 'number' ? pos.x : fallback[0];
    const y = typeof pos.y === 'number' ? pos.y : fallback[1];
    const z = typeof pos.z === 'number' ? pos.z : fallback[2];

    if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
      return [x, y, z];
    }
  }

  return fallback;
}



