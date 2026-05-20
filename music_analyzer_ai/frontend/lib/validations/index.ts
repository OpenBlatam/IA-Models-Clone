/**
 * Validation schemas exports.
 * Centralized export point for all Zod validation schemas.
 */

export * from './common';
export * from './music';

// Re-export array helper functions for convenience
export {
  arrayWithMinLength,
  arrayWithMaxLength,
  arrayWithLengthRange,
} from './common';
