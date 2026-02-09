/**
 * Hook for calculating smooth trajectory paths
 * @module robot-3d-view/hooks/use-trajectory
 */

import { useMemo } from 'react';
import * as THREE from 'three';
import type { Position } from '@/lib/api/types';
import type { Waypoint } from '../types';
import { smoothStep } from '../utils/math-utils';

/**
 * Options for trajectory calculation
 */
interface TrajectoryOptions {
  steps?: number;
  arcHeight?: number;
}

/**
 * Calculates a smooth trajectory path between two positions
 * 
 * @param currentPosition - Current robot position
 * @param targetPosition - Target robot position
 * @param options - Trajectory calculation options
 * @returns Array of waypoints forming a smooth curve
 * 
 * @example
 * ```tsx
 * const waypoints = useTrajectory(currentPos, targetPos, { steps: 50 });
 * ```
 */
export function useTrajectory(
  currentPosition: Position | null,
  targetPosition: Position | null,
  options: TrajectoryOptions = {}
): Waypoint[] {
  const { steps = 50, arcHeight = 0.3 } = options;

  return useMemo(() => {
    if (!currentPosition || !targetPosition) {
      return [];
    }

    const path: Waypoint[] = [];
    
    // Calculate intermediate control point for smooth arc
    const midX = (currentPosition.x + targetPosition.x) / 2;
    const midY = Math.max(currentPosition.y, targetPosition.y) + arcHeight;
    const midZ = (currentPosition.z + targetPosition.z) / 2;

    // Generate quadratic bezier curve points with smooth easing
    for (let i = 0; i <= steps; i++) {
      const rawT = i / steps;
      const t = smoothStep(rawT); // Apply smooth easing
      
      const x = (1 - t) * (1 - t) * currentPosition.x + 
                2 * (1 - t) * t * midX + 
                t * t * targetPosition.x;
      const y = (1 - t) * (1 - t) * currentPosition.y + 
                2 * (1 - t) * t * midY + 
                t * t * targetPosition.y;
      const z = (1 - t) * (1 - t) * currentPosition.z + 
                2 * (1 - t) * t * midZ + 
                t * t * targetPosition.z;
      path.push([x, y, z]);
    }

    return path;
  }, [currentPosition, targetPosition, steps, arcHeight]);
}

/**
 * Creates a THREE.js curve from waypoints
 * 
 * @param waypoints - Array of waypoints
 * @returns THREE.Curve or null if insufficient points
 * 
 * @example
 * ```tsx
 * const curve = useTrajectoryCurve(waypoints);
 * ```
 */
export function useTrajectoryCurve(waypoints: Waypoint[]): THREE.Curve<THREE.Vector3> | null {
  return useMemo(() => {
    if (waypoints.length < 2) return null;
    
    const points = waypoints.map((wp) => new THREE.Vector3(...wp));
    
    // Use CatmullRomCurve3 for smooth curves with 4+ points
    if (waypoints.length >= 4) {
      return new THREE.CatmullRomCurve3(points, false, 'centripetal');
    }
    
    // Use quadratic bezier for fewer points
    return new THREE.QuadraticBezierCurve3(
      points[0],
      points[Math.floor(points.length / 2)],
      points[points.length - 1]
    );
  }, [waypoints]);
}

