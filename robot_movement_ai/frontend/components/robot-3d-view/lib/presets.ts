/**
 * Configuration presets for Robot 3D View
 * 
 * Provides pre-configured settings for different use cases.
 * 
 * @module robot-3d-view/lib/presets
 */

import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * Preset types
 */
export type PresetType = 'minimal' | 'standard' | 'detailed' | 'performance' | 'quality';

/**
 * Preset configurations
 */
export const PRESETS: Record<PresetType, SceneConfig> = {
  minimal: {
    showStats: false,
    showGizmo: false,
    showStars: false,
    showWaypoints: false,
    showGrid: false,
    showObjects: false,
    autoRotate: false,
    gridSize: 5,
    cameraPreset: null,
  },
  standard: {
    showStats: false,
    showGizmo: true,
    showStars: false,
    showWaypoints: true,
    showGrid: true,
    showObjects: true,
    autoRotate: false,
    gridSize: 10,
    cameraPreset: null,
  },
  detailed: {
    showStats: true,
    showGizmo: true,
    showStars: true,
    showWaypoints: true,
    showGrid: true,
    showObjects: true,
    autoRotate: false,
    gridSize: 10,
    cameraPreset: null,
  },
  performance: {
    showStats: false,
    showGizmo: false,
    showStars: false,
    showWaypoints: false,
    showGrid: false,
    showObjects: false,
    autoRotate: false,
    gridSize: 5,
    cameraPreset: null,
  },
  quality: {
    showStats: true,
    showGizmo: true,
    showStars: true,
    showWaypoints: true,
    showGrid: true,
    showObjects: true,
    autoRotate: false,
    gridSize: 15,
    cameraPreset: null,
  },
};

/**
 * Gets a preset configuration
 * 
 * @param preset - Preset type
 * @returns Preset configuration
 */
export function getPreset(preset: PresetType): SceneConfig {
  return PRESETS[preset];
}

/**
 * Applies a preset to existing configuration
 * 
 * @param currentConfig - Current configuration
 * @param preset - Preset to apply
 * @returns Merged configuration
 */
export function applyPreset(
  currentConfig: SceneConfig,
  preset: PresetType
): SceneConfig {
  return {
    ...currentConfig,
    ...PRESETS[preset],
  };
}



