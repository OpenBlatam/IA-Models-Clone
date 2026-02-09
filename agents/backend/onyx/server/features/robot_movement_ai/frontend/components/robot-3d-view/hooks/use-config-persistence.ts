/**
 * Hook for persisting 3D view configuration
 * @module robot-3d-view/hooks/use-config-persistence
 */

import { useEffect, useCallback } from 'react';
import type { SceneConfig } from '../types';

const STORAGE_KEY = 'robot-3d-view-config';

/**
 * Hook for persisting and restoring 3D view configuration
 * 
 * Saves configuration to localStorage and restores it on mount.
 * 
 * @param config - Current configuration
 * @param updateConfig - Function to update configuration
 * 
 * @example
 * ```tsx
 * const { config, updateConfig } = use3DViewConfig();
 * useConfigPersistence(config, updateConfig);
 * ```
 */
export function useConfigPersistence(
  config: SceneConfig,
  updateConfig: (updates: Partial<SceneConfig>) => void
) {
  // Load configuration on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as Partial<SceneConfig>;
        updateConfig(parsed);
      }
    } catch (error) {
      console.warn('Failed to load 3D view configuration:', error);
    }
  }, [updateConfig]);

  // Save configuration on change
  useEffect(() => {
    if (typeof window === 'undefined') return;

    try {
      // Only save non-default values
      const toSave: Partial<SceneConfig> = {
        showStats: config.showStats,
        showGizmo: config.showGizmo,
        showStars: config.showStars,
        showWaypoints: config.showWaypoints,
        showGrid: config.showGrid,
        showObjects: config.showObjects,
        autoRotate: config.autoRotate,
        gridSize: config.gridSize,
        cameraPreset: config.cameraPreset,
      };

      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    } catch (error) {
      console.warn('Failed to save 3D view configuration:', error);
    }
  }, [config]);

  const clearConfig = useCallback(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.warn('Failed to clear 3D view configuration:', error);
    }
  }, []);

  return { clearConfig };
}



