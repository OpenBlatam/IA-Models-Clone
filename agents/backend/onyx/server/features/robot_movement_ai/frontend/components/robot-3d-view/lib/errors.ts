/**
 * Custom error types for Robot 3D View
 * @module robot-3d-view/lib/errors
 */

/**
 * Base error class for Robot 3D View errors
 */
export class Robot3DViewError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'Robot3DViewError';
    Object.setPrototypeOf(this, Robot3DViewError.prototype);
  }
}

/**
 * Position validation error
 */
export class PositionValidationError extends Robot3DViewError {
  constructor(message: string, cause?: unknown) {
    super(message, 'POSITION_VALIDATION_ERROR', cause);
    this.name = 'PositionValidationError';
    Object.setPrototypeOf(this, PositionValidationError.prototype);
  }
}

/**
 * Configuration error
 */
export class ConfigurationError extends Robot3DViewError {
  constructor(message: string, cause?: unknown) {
    super(message, 'CONFIGURATION_ERROR', cause);
    this.name = 'ConfigurationError';
    Object.setPrototypeOf(this, ConfigurationError.prototype);
  }
}

/**
 * Rendering error
 */
export class RenderingError extends Robot3DViewError {
  constructor(message: string, cause?: unknown) {
    super(message, 'RENDERING_ERROR', cause);
    this.name = 'RenderingError';
    Object.setPrototypeOf(this, RenderingError.prototype);
  }
}

/**
 * Trajectory calculation error
 */
export class TrajectoryError extends Robot3DViewError {
  constructor(message: string, cause?: unknown) {
    super(message, 'TRAJECTORY_ERROR', cause);
    this.name = 'TrajectoryError';
    Object.setPrototypeOf(this, TrajectoryError.prototype);
  }
}

/**
 * Error factory for creating typed errors
 */
export const createError = {
  position: (message: string, cause?: unknown) =>
    new PositionValidationError(message, cause),
  configuration: (message: string, cause?: unknown) =>
    new ConfigurationError(message, cause),
  rendering: (message: string, cause?: unknown) =>
    new RenderingError(message, cause),
  trajectory: (message: string, cause?: unknown) =>
    new TrajectoryError(message, cause),
};

/**
 * Type guard for Robot3DViewError
 */
export function isRobot3DViewError(error: unknown): error is Robot3DViewError {
  return error instanceof Robot3DViewError;
}



