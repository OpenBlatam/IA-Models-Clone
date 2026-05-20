/**
 * Validation constants
 * 
 * Centralized validation limits and rules
 */

/**
 * Validation limits for agent fields
 */
export const VALIDATION_LIMITS = {
  /** Minimum name length */
  MIN_NAME_LENGTH: 3,
  /** Maximum name length */
  MAX_NAME_LENGTH: 100,
  /** Maximum description length */
  MAX_DESCRIPTION_LENGTH: 500,
  /** Minimum frequency in seconds */
  MIN_FREQUENCY: 60,
  /** Maximum frequency in seconds */
  MAX_FREQUENCY: 86400 * 30, // 30 days
  /** Maximum parameters JSON size in characters */
  MAX_PARAMETERS_SIZE: 10000,
} as const;

/**
 * Validation patterns
 */
export const VALIDATION_PATTERNS = {
  /** Valid agent name pattern (alphanumeric, spaces, hyphens, underscores) */
  AGENT_NAME: /^[a-zA-Z0-9\s\-_]+$/,
  /** Valid JSON pattern */
  JSON: /^[\s\S]*$/,
} as const;

/**
 * Validation error messages
 */
export const VALIDATION_MESSAGES = {
  NAME_REQUIRED: "El nombre es requerido",
  NAME_TOO_SHORT: `El nombre debe tener al menos ${VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres`,
  NAME_TOO_LONG: `El nombre no puede exceder ${VALIDATION_LIMITS.MAX_NAME_LENGTH} caracteres`,
  NAME_INVALID: "El nombre contiene caracteres inválidos",
  DESCRIPTION_REQUIRED: "La descripción es requerida",
  DESCRIPTION_TOO_LONG: `La descripción no puede exceder ${VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH} caracteres`,
  FREQUENCY_REQUIRED: "La frecuencia es requerida",
  FREQUENCY_TOO_LOW: `La frecuencia mínima es ${VALIDATION_LIMITS.MIN_FREQUENCY} segundos`,
  FREQUENCY_TOO_HIGH: `La frecuencia máxima es ${VALIDATION_LIMITS.MAX_FREQUENCY} segundos`,
  FREQUENCY_INVALID: "La frecuencia debe ser un número positivo",
  PARAMETERS_INVALID_JSON: "Los parámetros deben ser un JSON válido",
  PARAMETERS_TOO_LARGE: `Los parámetros no pueden exceder ${VALIDATION_LIMITS.MAX_PARAMETERS_SIZE} caracteres`,
  PARAMETERS_NOT_OBJECT: "Los parámetros deben ser un objeto JSON",
} as const;




