/**
 * Tests for validation schemas
 * @module robot-3d-view/__tests__/validation-schemas
 */

import { describe, it, expect } from '@jest/globals';
import {
  position3DSchema,
  sceneConfigSchema,
  robot3DViewPropsSchema,
  safeParsePosition3D,
  safeParseSceneConfig,
} from '../schemas/validation-schemas';

describe('validation-schemas', () => {
  describe('position3DSchema', () => {
    it('should validate valid position tuple', () => {
      const position: [number, number, number] = [1, 2, 3];
      const result = position3DSchema.safeParse(position);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual([1, 2, 3]);
      }
    });

    it('should reject invalid position with NaN', () => {
      const position = [1, NaN, 3] as [number, number, number];
      const result = position3DSchema.safeParse(position);
      expect(result.success).toBe(false);
    });

    it('should reject position with wrong length', () => {
      const position = [1, 2] as unknown as [number, number, number];
      const result = position3DSchema.safeParse(position);
      expect(result.success).toBe(false);
    });
  });

  describe('sceneConfigSchema', () => {
    it('should validate valid scene config', () => {
      const config = {
        showStats: true,
        showGizmo: false,
        showStars: true,
        showWaypoints: false,
        showGrid: true,
        showObjects: false,
        autoRotate: true,
        gridSize: 10,
        cameraPreset: 'front' as const,
      };
      const result = sceneConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
    });

    it('should apply default values', () => {
      const config = {
        showStats: true,
        showGizmo: false,
        showStars: true,
        showWaypoints: false,
        showGrid: true,
        showObjects: false,
        autoRotate: true,
      };
      const result = sceneConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.gridSize).toBe(10);
        expect(result.data.cameraPreset).toBe(null);
      }
    });

    it('should reject invalid gridSize', () => {
      const config = {
        showStats: true,
        showGizmo: false,
        showStars: true,
        showWaypoints: false,
        showGrid: true,
        showObjects: false,
        autoRotate: true,
        gridSize: 100, // Out of range
        cameraPreset: null,
      };
      const result = sceneConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });
  });

  describe('robot3DViewPropsSchema', () => {
    it('should validate valid props', () => {
      const props = {
        fullscreen: true,
        className: 'custom-class',
      };
      const result = robot3DViewPropsSchema.safeParse(props);
      expect(result.success).toBe(true);
    });

    it('should apply default values', () => {
      const props = {};
      const result = robot3DViewPropsSchema.safeParse(props);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.fullscreen).toBe(false);
        expect(result.data.className).toBe('');
      }
    });
  });

  describe('safeParse utilities', () => {
    it('safeParsePosition3D should return success for valid position', () => {
      const position: [number, number, number] = [1, 2, 3];
      const result = safeParsePosition3D(position);
      expect(result.success).toBe(true);
    });

    it('safeParseSceneConfig should return success for valid config', () => {
      const config = {
        showStats: true,
        showGizmo: false,
        showStars: true,
        showWaypoints: false,
        showGrid: true,
        showObjects: false,
        autoRotate: true,
        gridSize: 10,
        cameraPreset: null,
      };
      const result = safeParseSceneConfig(config);
      expect(result.success).toBe(true);
    });
  });
});



