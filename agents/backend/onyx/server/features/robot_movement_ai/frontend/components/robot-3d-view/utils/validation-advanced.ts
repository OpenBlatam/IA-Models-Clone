/**
 * Advanced validation utilities
 * @module robot-3d-view/utils/validation-advanced
 */

import { z } from 'zod';
import type { Position3D, SceneConfig } from '../schemas/validation-schemas';

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

/**
 * Validation error
 */
export interface ValidationError {
  field: string;
  message: string;
  code: string;
  path?: (string | number)[];
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  field: string;
  message: string;
  code: string;
}

/**
 * Validates position with advanced checks
 */
export function validatePositionAdvanced(
  position: Position3D,
  bounds?: {
    min?: Position3D;
    max?: Position3D;
  }
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  // Check if position is array
  if (!Array.isArray(position) || position.length !== 3) {
    errors.push({
      field: 'position',
      message: 'Position must be an array of 3 numbers',
      code: 'INVALID_FORMAT',
    });
    return { valid: false, errors, warnings };
  }

  // Check if all values are numbers
  position.forEach((value, index) => {
    if (typeof value !== 'number' || isNaN(value)) {
      errors.push({
        field: `position[${index}]`,
        message: `Position coordinate ${index} must be a number`,
        code: 'INVALID_TYPE',
        path: [index],
      });
    }
  });

  // Check bounds
  if (bounds) {
    if (bounds.min) {
      position.forEach((value, index) => {
        if (value < bounds.min![index]) {
          errors.push({
            field: `position[${index}]`,
            message: `Position coordinate ${index} is below minimum (${bounds.min![index]})`,
            code: 'OUT_OF_BOUNDS',
            path: [index],
          });
        }
      });
    }

    if (bounds.max) {
      position.forEach((value, index) => {
        if (value > bounds.max![index]) {
          errors.push({
            field: `position[${index}]`,
            message: `Position coordinate ${index} is above maximum (${bounds.max![index]})`,
            code: 'OUT_OF_BOUNDS',
            path: [index],
          });
        }
      });
    }
  }

  // Check for extreme values (warnings)
  position.forEach((value, index) => {
    if (Math.abs(value) > 1000) {
      warnings.push({
        field: `position[${index}]`,
        message: `Position coordinate ${index} has an extreme value (${value})`,
        code: 'EXTREME_VALUE',
      });
    }
  });

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validates scene configuration with advanced checks
 */
export function validateSceneConfigAdvanced(
  config: unknown
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (typeof config !== 'object' || config === null) {
    errors.push({
      field: 'config',
      message: 'Configuration must be an object',
      code: 'INVALID_TYPE',
    });
    return { valid: false, errors, warnings };
  }

  const cfg = config as Record<string, unknown>;

  // Validate boolean fields
  const booleanFields = [
    'showStats',
    'showGizmo',
    'showStars',
    'showWaypoints',
    'showGrid',
    'showObjects',
    'autoRotate',
  ];

  booleanFields.forEach((field) => {
    if (field in cfg && typeof cfg[field] !== 'boolean') {
      errors.push({
        field,
        message: `${field} must be a boolean`,
        code: 'INVALID_TYPE',
      });
    }
  });

  // Validate camera preset
  if ('cameraPreset' in cfg) {
    const validPresets = ['front', 'top', 'side', 'iso', null];
    if (!validPresets.includes(cfg.cameraPreset as string | null)) {
      errors.push({
        field: 'cameraPreset',
        message: `Invalid camera preset. Must be one of: ${validPresets.join(', ')}`,
        code: 'INVALID_VALUE',
      });
    }
  }

  // Validate grid size
  if ('gridSize' in cfg) {
    const gridSize = cfg.gridSize;
    if (typeof gridSize !== 'number' || gridSize < 0) {
      errors.push({
        field: 'gridSize',
        message: 'Grid size must be a positive number',
        code: 'INVALID_VALUE',
      });
    } else if (gridSize > 100) {
      warnings.push({
        field: 'gridSize',
        message: 'Grid size is very large, may impact performance',
        code: 'PERFORMANCE_WARNING',
      });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validates trajectory with advanced checks
 */
export function validateTrajectoryAdvanced(
  trajectory: Position3D[]
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (!Array.isArray(trajectory)) {
    errors.push({
      field: 'trajectory',
      message: 'Trajectory must be an array',
      code: 'INVALID_TYPE',
    });
    return { valid: false, errors, warnings };
  }

  if (trajectory.length === 0) {
    warnings.push({
      field: 'trajectory',
      message: 'Trajectory is empty',
      code: 'EMPTY_TRAJECTORY',
    });
  }

  trajectory.forEach((point, index) => {
    const pointValidation = validatePositionAdvanced(point);
    if (!pointValidation.valid) {
      pointValidation.errors.forEach((error) => {
        errors.push({
          ...error,
          field: `trajectory[${index}].${error.field}`,
          path: [index, ...(error.path || [])],
        });
      });
    }
  });

  // Check for duplicate points
  const pointStrings = trajectory.map((p) => p.join(','));
  const duplicates = pointStrings.filter(
    (p, i) => pointStrings.indexOf(p) !== i
  );
  if (duplicates.length > 0) {
    warnings.push({
      field: 'trajectory',
      message: `Found ${duplicates.length} duplicate points in trajectory`,
      code: 'DUPLICATE_POINTS',
    });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Combines multiple validation results
 */
export function combineValidationResults(
  ...results: ValidationResult[]
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  results.forEach((result) => {
    errors.push(...result.errors);
    warnings.push(...result.warnings);
  });

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}



