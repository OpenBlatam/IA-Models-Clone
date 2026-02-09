/**
 * Tests for error handling
 * @module robot-3d-view/__tests__/errors.test
 */

import { describe, it, expect } from '@jest/globals';
import {
  Robot3DViewError,
  PositionValidationError,
  ConfigurationError,
  RenderingError,
  TrajectoryError,
  createError,
  isRobot3DViewError,
} from '../lib/errors';

describe('errors', () => {
  describe('Robot3DViewError', () => {
    it('should create error with message and code', () => {
      const error = new Robot3DViewError('Test error', 'TEST_ERROR');
      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_ERROR');
      expect(error.name).toBe('Robot3DViewError');
    });

    it('should include cause if provided', () => {
      const cause = new Error('Original error');
      const error = new Robot3DViewError('Test error', 'TEST_ERROR', cause);
      expect(error.cause).toBe(cause);
    });
  });

  describe('PositionValidationError', () => {
    it('should create position validation error', () => {
      const error = new PositionValidationError('Invalid position');
      expect(error.message).toBe('Invalid position');
      expect(error.code).toBe('POSITION_VALIDATION_ERROR');
      expect(error.name).toBe('PositionValidationError');
    });
  });

  describe('ConfigurationError', () => {
    it('should create configuration error', () => {
      const error = new ConfigurationError('Invalid config');
      expect(error.message).toBe('Invalid config');
      expect(error.code).toBe('CONFIGURATION_ERROR');
      expect(error.name).toBe('ConfigurationError');
    });
  });

  describe('RenderingError', () => {
    it('should create rendering error', () => {
      const error = new RenderingError('Render failed');
      expect(error.message).toBe('Render failed');
      expect(error.code).toBe('RENDERING_ERROR');
      expect(error.name).toBe('RenderingError');
    });
  });

  describe('TrajectoryError', () => {
    it('should create trajectory error', () => {
      const error = new TrajectoryError('Trajectory failed');
      expect(error.message).toBe('Trajectory failed');
      expect(error.code).toBe('TRAJECTORY_ERROR');
      expect(error.name).toBe('TrajectoryError');
    });
  });

  describe('createError factory', () => {
    it('should create position error', () => {
      const error = createError.position('Invalid position');
      expect(error).toBeInstanceOf(PositionValidationError);
    });

    it('should create configuration error', () => {
      const error = createError.configuration('Invalid config');
      expect(error).toBeInstanceOf(ConfigurationError);
    });

    it('should create rendering error', () => {
      const error = createError.rendering('Render failed');
      expect(error).toBeInstanceOf(RenderingError);
    });

    it('should create trajectory error', () => {
      const error = createError.trajectory('Trajectory failed');
      expect(error).toBeInstanceOf(TrajectoryError);
    });
  });

  describe('isRobot3DViewError', () => {
    it('should return true for Robot3DViewError', () => {
      const error = new Robot3DViewError('Test', 'TEST');
      expect(isRobot3DViewError(error)).toBe(true);
    });

    it('should return true for PositionValidationError', () => {
      const error = new PositionValidationError('Test');
      expect(isRobot3DViewError(error)).toBe(true);
    });

    it('should return false for regular Error', () => {
      const error = new Error('Test');
      expect(isRobot3DViewError(error)).toBe(false);
    });

    it('should return false for non-error values', () => {
      expect(isRobot3DViewError(null)).toBe(false);
      expect(isRobot3DViewError('string')).toBe(false);
      expect(isRobot3DViewError(123)).toBe(false);
    });
  });
});



