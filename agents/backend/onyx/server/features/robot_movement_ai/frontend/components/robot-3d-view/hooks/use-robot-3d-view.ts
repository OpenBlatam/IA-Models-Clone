/**
 * Main hook for Robot 3D View
 * 
 * Consolidates all hooks and state management for the 3D view.
 * Provides a single interface for the component to use.
 * 
 * @module robot-3d-view/hooks/use-robot-3d-view
 */

import { useMemo, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { useTrajectory } from './use-trajectory';
import { use3DViewConfig } from './use-3d-view-config';
import { useViewportOptimization } from './use-viewport-optimization';
import { useConfigPersistence } from './use-config-persistence';
import { positionTo3D } from '../lib/position-utils';
import { logger } from '../utils/logger';
import type { Position3D } from '../schemas/validation-schemas';

/**
 * Return type for useRobot3DView hook
 */
export interface UseRobot3DViewReturn {
  // Positions
  currentPos: Position3D;
  targetPos: Position3D | null;
  trajectory: Position3D[];
  
  // Configuration
  config: ReturnType<typeof use3DViewConfig>['config'];
  viewportQuality: ReturnType<typeof useViewportOptimization>['quality'];
  
  // Actions
  toggleStats: () => void;
  toggleGizmo: () => void;
  toggleStars: () => void;
  toggleWaypoints: () => void;
  toggleGrid: () => void;
  toggleObjects: () => void;
  toggleAutoRotate: () => void;
  setCameraPreset: (preset: ReturnType<typeof use3DViewConfig>['config']['cameraPreset']) => void;
  
  // Status
  status: ReturnType<typeof useRobotStore>['status'];
}

/**
 * Main hook for Robot 3D View
 * 
 * Consolidates all hooks and provides a clean interface for the component.
 * 
 * @param fullscreen - Whether the view is in fullscreen mode
 * @returns All state and actions for the 3D view
 */
export function useRobot3DView(fullscreen: boolean): UseRobot3DViewReturn {
  const { status, currentPosition, targetPosition } = useRobotStore();
  
  const {
    config,
    updateConfig,
    toggleStats,
    toggleGizmo,
    toggleStars,
    toggleWaypoints,
    toggleGrid,
    toggleObjects,
    toggleAutoRotate,
    setCameraPreset,
  } = use3DViewConfig();

  const { quality: viewportQuality } = useViewportOptimization();
  useConfigPersistence(config, updateConfig);

  // Calculate trajectory
  const trajectory = useTrajectory(currentPosition, targetPosition);

  // Convert positions to 3D tuples with validation
  const currentPos: Position3D = useMemo(
    () => positionTo3D(currentPosition),
    [currentPosition]
  );

  const targetPos: Position3D | null = useMemo(() => {
    if (!targetPosition) return null;
    return positionTo3D(targetPosition);
  }, [targetPosition]);

  // Log component mount/unmount
  useEffect(() => {
    logger.debug('Robot3DView mounted', { fullscreen, viewportQuality });
    return () => {
      logger.debug('Robot3DView unmounted');
    };
  }, [fullscreen, viewportQuality]);

  return {
    currentPos,
    targetPos,
    trajectory,
    config,
    viewportQuality,
    toggleStats,
    toggleGizmo,
    toggleStars,
    toggleWaypoints,
    toggleGrid,
    toggleObjects,
    toggleAutoRotate,
    setCameraPreset,
    status,
  };
}



