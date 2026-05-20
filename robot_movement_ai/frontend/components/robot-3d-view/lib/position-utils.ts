/**
 * Position utility functions
 * @module robot-3d-view/lib/position-utils
 */

import { position3DSchema, safeParsePosition3D } from '../schemas/validation-schemas';
import { createError } from './errors';
import type { Position } from '@/lib/api/types';

/**
 * Default fallback position
 */
const DEFAULT_POSITION: [number, number, number] = [0, 0, 0];

/**
 * Position bounds for validation
 */
const POSITION_BOUNDS = {
  min: -1000,
  max: 1000,
} as const;

/**
 * Converts Position object to Position3D tuple with validation
 * 
 * @param position - Position object from store
 * @param fallback - Fallback position if invalid
 * @param throwOnError - Whether to throw on validation error (default: false)
 * @returns Valid Position3D tuple
 * @throws {PositionValidationError} If throwOnError is true and validation fails
 */
export function positionTo3D(
  position: Position | null | undefined,
  fallback: [number, number, number] = DEFAULT_POSITION,
  throwOnError = false
): [number, number, number] {
  // Early return for null/undefined
  if (!position) {
    if (throwOnError) {
      throw createError.position('Position is null or undefined');
    }
    return fallback;
  }

  const tuple: [number, number, number] = [position.x, position.y, position.z];
  const result = safeParsePosition3D(tuple);

  if (!result.success) {
    if (throwOnError) {
      throw createError.position('Invalid position format', result.error);
    }
    return fallback;
  }

  // Validate bounds
  const [x, y, z] = result.data;
  const isOutOfBounds =
    x < POSITION_BOUNDS.min || x > POSITION_BOUNDS.max ||
    y < POSITION_BOUNDS.min || y > POSITION_BOUNDS.max ||
    z < POSITION_BOUNDS.min || z > POSITION_BOUNDS.max;

  if (isOutOfBounds) {
    if (throwOnError) {
      throw createError.position(
        `Position out of bounds: [${x}, ${y}, ${z}]`
      );
    }
    return fallback;
  }

  return result.data;
}

/**
 * Validates if a position is within bounds
 * 
 * @param position - Position to validate
 * @param bounds - Optional custom bounds
 * @returns true if position is valid
 */
export function isValidPosition3D(
  position: [number, number, number],
  bounds = POSITION_BOUNDS
): boolean {
  const result = safeParsePosition3D(position);
  if (!result.success) {
    return false;
  }

  const [x, y, z] = result.data;
  return (
    x >= bounds.min && x <= bounds.max &&
    y >= bounds.min && y <= bounds.max &&
    z >= bounds.min && z <= bounds.max
  );
}

/**
 * Calculates distance between two positions
 * 
 * @param p1 - First position
 * @param p2 - Second position
 * @returns Euclidean distance
 */
export function distance3D(
  p1: [number, number, number],
  p2: [number, number, number]
): number {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const dz = p2[2] - p1[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

