/**
 * Trajectory Path 3D Component
 * @module robot-3d-view/objects/trajectory-path
 */

'use client';

import { useMemo, memo } from 'react';
import { Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { TrajectoryPathProps } from '../types';
import { useTrajectoryCurve } from '../hooks/use-trajectory';

/**
 * Improved trajectory with smooth curves and waypoints
 * 
 * Renders a smooth trajectory path between waypoints with optional
 * waypoint markers and labels.
 * 
 * @param props - Trajectory path component props
 * @returns Trajectory path 3D component
 * 
 * @example
 * ```tsx
 * <TrajectoryPath waypoints={[[0,0,0], [1,1,1], [2,2,2]]} showWaypoints />
 * ```
 */
export const TrajectoryPath = memo(({ waypoints, showWaypoints = true }: TrajectoryPathProps) => {
  const curve = useTrajectoryCurve(waypoints);

  const points = useMemo(() => {
    if (!curve) return [];
    return curve.getPoints(50);
  }, [curve]);

  if (!curve || points.length === 0) {
    return null;
  }

  return (
    <group>
      {/* Main trajectory line */}
      <Line points={points} color="#f59e0b" lineWidth={3} dashed={false} />
      
      {/* Waypoint markers */}
      {showWaypoints &&
        waypoints.map((wp, i) => (
          <group key={i}>
            <mesh position={wp}>
              <sphereGeometry args={[0.03, 8, 8]} />
              <meshStandardMaterial
                color="#f59e0b"
                emissive="#f59e0b"
                emissiveIntensity={0.8}
              />
            </mesh>
            <Html position={[0, 0.1, 0]} center>
              <div className="bg-orange-500/90 backdrop-blur-sm px-1.5 py-0.5 rounded text-[10px] font-semibold text-black whitespace-nowrap">
                {i + 1}
              </div>
            </Html>
          </group>
        ))}
    </group>
  );
});

TrajectoryPath.displayName = 'TrajectoryPath';



