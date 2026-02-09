/**
 * Zod validation schemas for Robot 3D View
 * @module robot-3d-view/schemas/validation-schemas
 */

import { z } from 'zod';

/**
 * Position3D schema - validates [x, y, z] tuple
 */
export const position3DSchema = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
]);

/**
 * Camera preset schema
 */
export const cameraPresetSchema = z.enum(['front', 'top', 'side', 'iso']).nullable();

/**
 * Robot status schema
 */
export const robotStatusSchema = z.enum(['idle', 'moving', 'error']);

/**
 * View configuration schema
 */
export const viewConfigSchema = z.object({
  showStats: z.boolean(),
  showGizmo: z.boolean(),
  showStars: z.boolean(),
  showWaypoints: z.boolean(),
  showGrid: z.boolean(),
  showObjects: z.boolean(),
  autoRotate: z.boolean(),
});

/**
 * Scene configuration schema
 */
export const sceneConfigSchema = viewConfigSchema.extend({
  gridSize: z.number().int().min(1).max(50).default(10),
  cameraPreset: cameraPresetSchema.default(null),
});

/**
 * Waypoint schema (array of Position3D)
 */
export const waypointSchema = position3DSchema;
export const waypointsSchema = z.array(waypointSchema);

/**
 * Material properties schema
 */
export const materialPropsSchema = z.object({
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/).optional(),
  metalness: z.number().min(0).max(1).optional(),
  roughness: z.number().min(0).max(1).optional(),
  emissive: z.string().regex(/^#[0-9A-Fa-f]{6}$/).optional(),
  emissiveIntensity: z.number().min(0).max(10).optional(),
  opacity: z.number().min(0).max(1).optional(),
  transparent: z.boolean().optional(),
});

/**
 * Animation configuration schema
 */
export const animationConfigSchema = z.object({
  speed: z.number().positive().optional(),
  rotationIntensity: z.number().min(0).max(1).optional(),
  floatIntensity: z.number().min(0).max(1).optional(),
});

/**
 * Base object props schema
 */
export const baseObjectPropsSchema = z.object({
  position: position3DSchema,
  rotation: position3DSchema.optional(),
});

/**
 * Robot3DView props schema
 */
export const robot3DViewPropsSchema = z.object({
  fullscreen: z.boolean().default(false),
  className: z.string().default(''),
});

/**
 * Type exports inferred from schemas
 */
export type Position3D = z.infer<typeof position3DSchema>;
export type CameraPreset = z.infer<typeof cameraPresetSchema>;
export type RobotStatus = z.infer<typeof robotStatusSchema>;
export type ViewConfig = z.infer<typeof viewConfigSchema>;
export type SceneConfig = z.infer<typeof sceneConfigSchema>;
export type Waypoint = z.infer<typeof waypointSchema>;
export type MaterialProps = z.infer<typeof materialPropsSchema>;
export type AnimationConfig = z.infer<typeof animationConfigSchema>;
export type BaseObjectProps = z.infer<typeof baseObjectPropsSchema>;
export type Robot3DViewProps = z.infer<typeof robot3DViewPropsSchema>;

/**
 * Safe parse utilities
 */
export const safeParsePosition3D = (data: unknown) => position3DSchema.safeParse(data);
export const safeParseSceneConfig = (data: unknown) => sceneConfigSchema.safeParse(data);
export const safeParseRobot3DViewProps = (data: unknown) => robot3DViewPropsSchema.safeParse(data);



