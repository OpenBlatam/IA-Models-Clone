/**
 * 3D Scene Component
 * 
 * Main scene orchestrator that composes all scene elements.
 * Refactored for better separation of concerns and maintainability.
 * 
 * @module robot-3d-view/scene/scene-3d
 */

'use client';

import { memo, useMemo } from 'react';
import { Grid, Environment, ContactShadows } from '../lib/drei-imports';
import { EnvironmentSetup } from './environment-setup';
import { CameraPreset } from './camera-preset';
import { LightingSetup } from './lighting-setup';
import { BackgroundSetup } from './background-setup';
import { SceneControls } from './scene-controls';
import { RobotSceneObjects } from './robot-scene-objects';
import { usePerformanceMonitor } from '../hooks/use-performance-monitor';
import { GRID_CONFIG } from '../constants';
import type { SceneConfig, Position3D, Waypoint } from '../types';

/**
 * Props for Scene3D component
 */
interface Scene3DProps {
  trajectory: Waypoint[];
  currentPos: Position3D;
  targetPos: Position3D | null;
  config: SceneConfig;
}

/**
 * Main 3D Scene Component
 * 
 * Orchestrates all scene elements with clear separation of concerns.
 * Each major aspect (lighting, background, controls, objects) is handled
 * by dedicated sub-components.
 * 
 * Features:
 * - Performance monitoring
 * - Optimized rendering with memoization
 * - Conditional rendering based on config
 * - Modular architecture for better maintainability
 * 
 * @param props - Scene configuration and data
 * @returns 3D scene component
 */
export const Scene3D = memo(({ trajectory, currentPos, targetPos, config }: Scene3DProps) => {
  // Performance monitoring (only when stats are shown)
  usePerformanceMonitor({
    enabled: config.showStats,
  });

  // Memoize grid args to prevent re-creation on every render
  const gridArgs = useMemo(
    () => [config.gridSize, config.gridSize] as [number, number],
    [config.gridSize]
  );

  return (
    <>
      {/* Background (sky and stars) */}
      <BackgroundSetup showStars={config.showStars} />

      {/* Lighting configuration */}
      <LightingSetup />

      {/* Environment and shadows */}
      <Environment preset="city" />
      <ContactShadows
        position={[0, -0.1, 0]}
        opacity={0.4}
        scale={10}
        blur={2}
        far={4.5}
      />

      {/* Grid */}
      {config.showGrid && (
        <Grid
          args={gridArgs}
          cellColor={GRID_CONFIG.cellColor}
          sectionColor={GRID_CONFIG.sectionColor}
          cellThickness={GRID_CONFIG.cellThickness}
          sectionThickness={GRID_CONFIG.sectionThickness}
          fadeDistance={GRID_CONFIG.fadeDistance}
          fadeStrength={GRID_CONFIG.fadeStrength}
        />
      )}

      {/* Robot-related objects */}
      <RobotSceneObjects
        currentPos={currentPos}
        targetPos={targetPos}
        trajectory={trajectory}
        config={config}
      />

      {/* Environment objects */}
      <EnvironmentSetup showObjects={config.showObjects} currentPos={currentPos} />

      {/* Coordinate axes helper */}
      <axesHelper args={[1]} />

      {/* Camera and navigation controls */}
      <SceneControls config={config} />

      {/* Camera preset */}
      {config.cameraPreset && (
        <CameraPreset preset={config.cameraPreset} target={currentPos} />
      )}
    </>
  );
});

Scene3D.displayName = 'Scene3D';
