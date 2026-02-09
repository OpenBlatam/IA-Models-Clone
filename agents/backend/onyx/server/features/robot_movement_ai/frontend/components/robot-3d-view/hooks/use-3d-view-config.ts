/**
 * Hook for managing 3D view configuration state
 * @module robot-3d-view/hooks/use-3d-view-config
 */

import { useState, useCallback } from 'react';
import type { SceneConfig, CameraPreset } from '../types';
import { DEFAULT_VIEW_CONFIG } from '../constants';

/**
 * Returns state and handlers for 3D view configuration
 * 
 * @returns Configuration state and update handlers
 * 
 * @example
 * ```tsx
 * const { config, toggleStats, setCameraPreset } = use3DViewConfig();
 * ```
 */
export function use3DViewConfig() {
  const [config, setConfig] = useState<SceneConfig>({
    ...DEFAULT_VIEW_CONFIG,
  });

  const updateConfig = useCallback((updates: Partial<SceneConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  const toggleStats = useCallback(() => {
    setConfig((prev) => ({ ...prev, showStats: !prev.showStats }));
  }, []);

  const toggleGizmo = useCallback(() => {
    setConfig((prev) => ({ ...prev, showGizmo: !prev.showGizmo }));
  }, []);

  const toggleStars = useCallback(() => {
    setConfig((prev) => ({ ...prev, showStars: !prev.showStars }));
  }, []);

  const toggleWaypoints = useCallback(() => {
    setConfig((prev) => ({ ...prev, showWaypoints: !prev.showWaypoints }));
  }, []);

  const toggleGrid = useCallback(() => {
    setConfig((prev) => ({ ...prev, showGrid: !prev.showGrid }));
  }, []);

  const toggleObjects = useCallback(() => {
    setConfig((prev) => ({ ...prev, showObjects: !prev.showObjects }));
  }, []);

  const toggleAutoRotate = useCallback(() => {
    setConfig((prev) => ({ ...prev, autoRotate: !prev.autoRotate }));
  }, []);

  const setCameraPreset = useCallback((preset: CameraPreset) => {
    setConfig((prev) => ({
      ...prev,
      cameraPreset: prev.cameraPreset === preset ? null : preset,
    }));
  }, []);

  const setGridSize = useCallback((size: number) => {
    setConfig((prev) => ({ ...prev, gridSize: size }));
  }, []);

  return {
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
    setGridSize,
  };
}



