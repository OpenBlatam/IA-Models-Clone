/**
 * Validation utilities for 3D View
 * @module robot-3d-view/utils/validations
 */

import type { Position3D, Waypoint } from '../types';

/**
 * Validates if a position is within reasonable bounds
 * 
 * @param position - Position to validate
 * @param bounds - Optional custom bounds [min, max]
 * @returns true if position is valid
 */
export function isValidPosition(
  position: Position3D,
  bounds: [number, number] = [-100, 100]
): boolean {
  const [min, max] = bounds;
  return position.every((coord) => {
    return typeof coord === 'number' && !isNaN(coord) && coord >= min && coord <= max;
  });
}

/**
 * Validates waypoints array
 * 
 * @param waypoints - Array of waypoints to validate
 * @returns true if all waypoints are valid
 */
export function isValidWaypoints(waypoints: Waypoint[]): boolean {
  if (!Array.isArray(waypoints) || waypoints.length === 0) {
    return false;
  }
  return waypoints.every((wp) => isValidPosition(wp));
}

/**
 * Clamps a value between min and max
 * 
 * @param value - Value to clamp
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clamped value
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Calculates distance between two positions
 * 
 * @param pos1 - First position
 * @param pos2 - Second position
 * @returns Euclidean distance
 */
export function calculateDistance(pos1: Position3D, pos2: Position3D): number {
  const dx = pos2[0] - pos1[0];
  const dy = pos2[1] - pos1[1];
  const dz = pos2[2] - pos1[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}



