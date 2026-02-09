/**
 * Tests for position utilities
 * @module robot-3d-view/__tests__/position-utils
 */

import { describe, it, expect } from '@jest/globals';
import { positionTo3D, isValidPosition3D, distance3D } from '../lib/position-utils';
import { createError } from '../lib/errors';

describe('position-utils', () => {
  describe('positionTo3D', () => {
    it('should convert valid position to Position3D', () => {
      const position = { x: 1, y: 2, z: 3 };
      const result = positionTo3D(position);
      expect(result).toEqual([1, 2, 3]);
    });

    it('should return fallback for null position', () => {
      const result = positionTo3D(null);
      expect(result).toEqual([0, 0, 0]);
    });

    it('should return fallback for undefined position', () => {
      const result = positionTo3D(undefined);
      expect(result).toEqual([0, 0, 0]);
    });

    it('should return custom fallback when provided', () => {
      const fallback: [number, number, number] = [10, 20, 30];
      const result = positionTo3D(null, fallback);
      expect(result).toEqual(fallback);
    });

    it('should return fallback for out-of-bounds position', () => {
      const position = { x: 2000, y: 0, z: 0 };
      const result = positionTo3D(position);
      expect(result).toEqual([0, 0, 0]);
    });

    it('should throw error when throwOnError is true and position is invalid', () => {
      const position = { x: 2000, y: 0, z: 0 };
      expect(() => positionTo3D(position, [0, 0, 0], true)).toThrow();
    });

    it('should throw PositionValidationError for invalid position', () => {
      const position = { x: NaN, y: 0, z: 0 };
      expect(() => positionTo3D(position, [0, 0, 0], true)).toThrow(
        createError.position('Invalid position format')
      );
    });
  });

  describe('isValidPosition3D', () => {
    it('should return true for valid position', () => {
      const position: [number, number, number] = [1, 2, 3];
      expect(isValidPosition3D(position)).toBe(true);
    });

    it('should return false for out-of-bounds position', () => {
      const position: [number, number, number] = [2000, 0, 0];
      expect(isValidPosition3D(position)).toBe(false);
    });

    it('should return false for invalid tuple length', () => {
      const position = [1, 2] as unknown as [number, number, number];
      expect(isValidPosition3D(position)).toBe(false);
    });
  });

  describe('distance3D', () => {
    it('should calculate distance between two positions', () => {
      const p1: [number, number, number] = [0, 0, 0];
      const p2: [number, number, number] = [3, 4, 0];
      expect(distance3D(p1, p2)).toBe(5);
    });

    it('should calculate 3D distance correctly', () => {
      const p1: [number, number, number] = [0, 0, 0];
      const p2: [number, number, number] = [1, 1, 1];
      const expected = Math.sqrt(3);
      expect(distance3D(p1, p2)).toBeCloseTo(expected);
    });

    it('should return 0 for identical positions', () => {
      const p1: [number, number, number] = [1, 2, 3];
      const p2: [number, number, number] = [1, 2, 3];
      expect(distance3D(p1, p2)).toBe(0);
    });
  });
});



