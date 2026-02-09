/**
 * Math utility functions for 3D calculations
 * @module robot-3d-view/utils/math-utils
 */

import type { Position3D } from '../types';

/**
 * Calculates Euclidean distance between two 3D points
 * 
 * @param p1 - First point
 * @param p2 - Second point
 * @returns Distance
 */
export function distance3D(p1: Position3D, p2: Position3D): number {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const dz = p2[2] - p1[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Linear interpolation between two values
 * 
 * @param start - Start value
 * @param end - End value
 * @param t - Interpolation factor (0-1)
 * @returns Interpolated value
 */
export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

/**
 * Linear interpolation for 3D positions
 * 
 * @param start - Start position
 * @param end - End position
 * @param t - Interpolation factor (0-1)
 * @returns Interpolated position
 */
export function lerp3D(start: Position3D, end: Position3D, t: number): Position3D {
  return [
    lerp(start[0], end[0], t),
    lerp(start[1], end[1], t),
    lerp(start[2], end[2], t),
  ];
}

/**
 * Normalizes a 3D vector
 * 
 * @param vector - Vector to normalize
 * @returns Normalized vector
 */
export function normalize3D(vector: Position3D): Position3D {
  const length = Math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2);
  if (length === 0) return [0, 0, 0];
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/**
 * Calculates midpoint between two positions
 * 
 * @param p1 - First position
 * @param p2 - Second position
 * @returns Midpoint
 */
export function midpoint3D(p1: Position3D, p2: Position3D): Position3D {
  return [
    (p1[0] + p2[0]) / 2,
    (p1[1] + p2[1]) / 2,
    (p1[2] + p2[2]) / 2,
  ];
}

/**
 * Smooth step interpolation (ease in-out)
 * 
 * @param t - Interpolation factor (0-1)
 * @returns Smoothed value
 */
export function smoothStep(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Ease in-out cubic interpolation
 * 
 * @param t - Interpolation factor (0-1)
 * @returns Eased value
 */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}



