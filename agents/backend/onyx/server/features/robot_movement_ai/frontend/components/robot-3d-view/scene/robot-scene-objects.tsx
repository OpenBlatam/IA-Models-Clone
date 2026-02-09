/**
 * Robot Scene Objects Component
 * 
 * Separated robot-related objects for better organization.
 * 
 * @module robot-3d-view/scene/robot-scene-objects
 */

'use client';

import { memo } from 'react';
import { RobotArm } from '../objects/robot-arm';
import { TargetMarker } from '../objects/target-marker';
import { TrajectoryPath } from '../objects/trajectory-path';
import { ParticleSystem } from '../effects/particle-system';
import { GlowEffect } from '../effects/glow-effect';
import type { SceneConfig, Position3D, Waypoint } from '../types';

/**
 * Props for RobotSceneObjects component
 */
interface RobotSceneObjectsProps {
  currentPos: Position3D;
  targetPos: Position3D | null;
  trajectory: Waypoint[];
  config: SceneConfig;
}

/**
 * Robot Scene Objects Component
 * 
 * Renders all robot-related objects (arm, target, trajectory, effects).
 * 
 * @param props - Robot scene objects configuration
 * @returns Robot scene objects component
 */
export const RobotSceneObjects = memo(({
  currentPos,
  targetPos,
  trajectory,
  config,
}: RobotSceneObjectsProps) => {
  return (
    <>
      {/* Particle effects around robot */}
      <ParticleSystem
        position={currentPos}
        count={20}
        color="#0ea5e9"
        speed={0.4}
        size={2}
        scale={[2, 2, 2]}
        enabled={config.showObjects}
      />

      {/* Glow effect on robot */}
      <GlowEffect
        position={currentPos}
        color="#0ea5e9"
        intensity={0.5}
        size={0.15}
        pulseSpeed={2}
        enabled={config.showObjects}
      />

      {/* Robot Arm */}
      <RobotArm position={currentPos} />

      {/* Target Position */}
      {targetPos && <TargetMarker position={targetPos} />}

      {/* Trajectory Path */}
      {trajectory.length > 0 && (
        <TrajectoryPath
          waypoints={trajectory}
          showWaypoints={config.showWaypoints}
        />
      )}
    </>
  );
});

RobotSceneObjects.displayName = 'RobotSceneObjects';



