/**
 * Robot-specific utilities
 */

import { Position } from '@/lib/api/types';
import { distance3D, clamp } from './math';
import { ROBOT_CONFIG } from './constants';

// Validate position
export function validatePosition(position: Position): boolean {
  const { MIN_POSITION, MAX_POSITION } = ROBOT_CONFIG;
  
  return (
    position.x >= MIN_POSITION &&
    position.x <= MAX_POSITION &&
    position.y >= MIN_POSITION &&
    position.y <= MAX_POSITION &&
    position.z >= MIN_POSITION &&
    position.z <= MAX_POSITION
  );
}

// Clamp position to valid range
export function clampPosition(position: Position): Position {
  const { MIN_POSITION, MAX_POSITION } = ROBOT_CONFIG;
  
  return {
    x: clamp(position.x, MIN_POSITION, MAX_POSITION),
    y: clamp(position.y, MIN_POSITION, MAX_POSITION),
    z: clamp(position.z, MIN_POSITION, MAX_POSITION),
  };
}

// Calculate distance between two positions
export function calculateDistance(pos1: Position, pos2: Position): number {
  return distance3D(pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z);
}

// Calculate movement time (simple estimation)
export function calculateMovementTime(
  from: Position,
  to: Position,
  speed: number = ROBOT_CONFIG.MOVEMENT_SPEED
): number {
  const distance = calculateDistance(from, to);
  return distance / speed; // seconds
}

// Interpolate between two positions
export function interpolatePosition(
  from: Position,
  to: Position,
  t: number
): Position {
  return {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
    z: from.z + (to.z - from.z) * t,
  };
}

// Check if position is home
export function isHomePosition(position: Position): boolean {
  return position.x === 0 && position.y === 0 && position.z === 0;
}

// Format position for display
export function formatPosition(position: Position, decimals: number = 3): string {
  return `(${position.x.toFixed(decimals)}, ${position.y.toFixed(decimals)}, ${position.z.toFixed(decimals)})`;
}

// Parse position from string
export function parsePosition(str: string): Position | null {
  try {
    // Try to parse formats like "(1, 2, 3)" or "1, 2, 3"
    const cleaned = str.replace(/[()]/g, '');
    const parts = cleaned.split(',').map((s) => parseFloat(s.trim()));
    
    if (parts.length === 3 && parts.every((n) => !isNaN(n))) {
      return {
        x: parts[0],
        y: parts[1],
        z: parts[2],
      };
    }
    
    return null;
  } catch {
    return null;
  }
}

// Generate random position
export function randomPosition(): Position {
  const { MIN_POSITION, MAX_POSITION } = ROBOT_CONFIG;
  
  return {
    x: Math.random() * (MAX_POSITION - MIN_POSITION) + MIN_POSITION,
    y: Math.random() * (MAX_POSITION - MIN_POSITION) + MIN_POSITION,
    z: Math.random() * (MAX_POSITION - MIN_POSITION) + MIN_POSITION,
  };
}



