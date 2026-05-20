/**
 * Validation utilities barrel export
 */

// Zod schemas and validators
export * from "./zod-schemas";
export * from "./zod-validator";
export * from "./constants";

// Legacy validation functions (kept for backward compatibility)
export {
  validateName,
  validateDescription,
  validateFrequency,
  validateJSON,
  validateRequired,
  parseJSON,
  VALIDATION_LIMITS,
  type ValidationResult,
} from "../validation";

