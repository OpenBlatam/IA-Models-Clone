/**
 * Constants for Robot 3D View
 * @module robot-3d-view/constants
 */

import type { Position3D, CameraPreset } from './types';

/**
 * Default camera positions for presets
 */
export const CAMERA_PRESETS: Record<NonNullable<CameraPreset>, { position: Position3D; target: Position3D }> = {
  front: {
    position: [0, 0, 5],
    target: [0, 0, 0],
  },
  top: {
    position: [0, 5, 0],
    target: [0, 0, 0],
  },
  side: {
    position: [5, 0, 0],
    target: [0, 0, 0],
  },
  iso: {
    position: [3, 3, 3],
    target: [0, 0, 0],
  },
};

/**
 * Default view configuration
 */
export const DEFAULT_VIEW_CONFIG = {
  showStats: false,
  showGizmo: true,
  showStars: false,
  showWaypoints: true,
  showGrid: true,
  showObjects: true,
  autoRotate: false,
  gridSize: 10,
  cameraPreset: null as CameraPreset,
} as const;

/**
 * Material constants
 */
export const MATERIALS = {
  robot: {
    base: { color: '#0ea5e9', metalness: 0.8, roughness: 0.2 },
    link: { color: '#0284c7', metalness: 0.7, roughness: 0.3 },
    joint: { color: '#0369a1', metalness: 0.9, roughness: 0.1 },
    effector: { color: '#10b981', metalness: 0.5, roughness: 0.4 },
  },
  target: {
    color: '#f59e0b',
    emissive: '#f59e0b',
    emissiveIntensity: 1.5,
  },
  obstacle: {
    color: '#ef4444',
    opacity: 0.5,
    emissive: '#ef4444',
    emissiveIntensity: 0.2,
  },
} as const;

/**
 * Animation constants
 */
export const ANIMATIONS = {
  robotArm: {
    speed: 1.5,
    rotationIntensity: 0.2,
    floatIntensity: 0.3,
  },
  targetMarker: {
    speed: 2,
    rotationIntensity: 0.5,
    floatIntensity: 0.5,
  },
} as const;

/**
 * Grid configuration
 */
export const GRID_CONFIG = {
  cellColor: '#374151',
  sectionColor: '#1f2937',
  cellThickness: 0.5,
  sectionThickness: 1,
  fadeDistance: 20,
  fadeStrength: 1,
} as const;

/**
 * Lighting configuration
 */
export const LIGHTING = {
  ambient: { intensity: 0.4 },
  directional: {
    main: { position: [10, 10, 5] as Position3D, intensity: 1.2 },
    secondary: { position: [-10, 5, -5] as Position3D, intensity: 0.5 },
  },
  point: { position: [0, 10, 0] as Position3D, intensity: 0.3 },
  spot: {
    position: [5, 10, 5] as Position3D,
    angle: 0.3,
    penumbra: 1,
    intensity: 0.5,
  },
} as const;

/**
 * Canvas configuration
 */
export const CANVAS_CONFIG = {
  camera: { position: [3, 3, 3] as Position3D, fov: 50 },
  gl: {
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance' as const,
  },
  dpr: [1, 2] as [number, number],
  shadows: true,
} as const;



