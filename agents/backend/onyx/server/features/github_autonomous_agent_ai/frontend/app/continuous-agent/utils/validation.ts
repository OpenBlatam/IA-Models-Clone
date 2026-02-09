/**
 * Validation utilities for Continuous Agent forms
 * 
 * Provides type-safe validation functions with consistent error messages
 */

import { FORM_DEFAULTS } from "../constants";

/**
 * Result of a validation operation
 */
export type ValidationResult = {
  /** Whether the value is valid */
  readonly isValid: boolean;
  /** Error message if invalid, null if valid */
  readonly error: string | null;
};

const MIN_NAME_LENGTH = 3;
const MAX_NAME_LENGTH = 100;
const MAX_DESCRIPTION_LENGTH = 500;

/**
 * Validates agent name
 * @param name - The name to validate
 * @returns Validation result
 */
export const validateName = (name: string): ValidationResult => {
  if (typeof name !== "string") {
    return {
      isValid: false,
      error: "El nombre debe ser un texto",
    };
  }

  const trimmed = name.trim();

  if (!trimmed) {
    return {
      isValid: false,
      error: "El nombre es requerido",
    };
  }

  if (trimmed.length < MIN_NAME_LENGTH) {
    return {
      isValid: false,
      error: `El nombre debe tener al menos ${MIN_NAME_LENGTH} caracteres`,
    };
  }

  if (trimmed.length > MAX_NAME_LENGTH) {
    return {
      isValid: false,
      error: `El nombre no puede exceder ${MAX_NAME_LENGTH} caracteres`,
    };
  }

  return { isValid: true, error: null };
};

/**
 * Validates agent description
 * @param description - The description to validate
 * @returns Validation result
 */
export const validateDescription = (description: string): ValidationResult => {
  if (typeof description !== "string") {
    return {
      isValid: false,
      error: "La descripción debe ser un texto",
    };
  }

  const trimmed = description.trim();

  if (!trimmed) {
    return {
      isValid: false,
      error: "La descripción es requerida",
    };
  }

  if (trimmed.length > MAX_DESCRIPTION_LENGTH) {
    return {
      isValid: false,
      error: `La descripción no puede exceder ${MAX_DESCRIPTION_LENGTH} caracteres`,
    };
  }

  return { isValid: true, error: null };
};

/**
 * Validates execution frequency
 * @param frequency - The frequency in seconds
 * @returns Validation result
 */
export const validateFrequency = (frequency: number): ValidationResult => {
  if (typeof frequency !== "number" || isNaN(frequency)) {
    return {
      isValid: false,
      error: "La frecuencia debe ser un número",
    };
  }

  if (frequency < FORM_DEFAULTS.MIN_FREQUENCY) {
    return {
      isValid: false,
      error: `La frecuencia mínima es ${FORM_DEFAULTS.MIN_FREQUENCY} segundos`,
    };
  }

  if (!Number.isFinite(frequency) || frequency <= 0) {
    return {
      isValid: false,
      error: "La frecuencia debe ser un número positivo",
    };
  }

  return { isValid: true, error: null };
};

/**
 * Validates JSON string
 * @param jsonString - The JSON string to validate
 * @returns Validation result
 */
export const validateJSON = (jsonString: string): ValidationResult => {
  if (typeof jsonString !== "string") {
    return {
      isValid: false,
      error: "Los parámetros deben ser un texto JSON",
    };
  }

  if (!jsonString.trim()) {
    return { isValid: true, error: null };
  }

  try {
    JSON.parse(jsonString);
    return { isValid: true, error: null };
  } catch (error) {
    const message = error instanceof Error ? error.message : "JSON inválido";
    return {
      isValid: false,
      error: `Los parámetros deben ser un JSON válido: ${message}`,
    };
  }
};

/**
 * Validates that a required field is not empty
 * @param value - The value to validate
 * @returns Validation result
 */
export const validateRequired = (value: string): ValidationResult => {
  if (typeof value !== "string") {
    return {
      isValid: false,
      error: "Este campo es requerido",
    };
  }

  if (!value.trim()) {
    return {
      isValid: false,
      error: "Este campo es requerido",
    };
  }

  return { isValid: true, error: null };
};

/**
 * Safely parses a JSON string
 * @param jsonString - The JSON string to parse
 * @returns Parsed object or empty object if string is empty
 * @throws Error if JSON is invalid
 */
export const parseJSON = (jsonString: string): Record<string, unknown> => {
  if (typeof jsonString !== "string") {
    throw new Error("Los parámetros deben ser un texto JSON");
  }

  if (!jsonString.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(jsonString);
    
    // Ensure result is an object
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("Los parámetros deben ser un objeto JSON");
    }
    
    return parsed;
  } catch (error) {
    const message = error instanceof Error ? error.message : "JSON inválido";
    throw new Error(`Los parámetros deben ser un JSON válido: ${message}`);
  }
};

const MAX_GOAL_LENGTH = 10000; // Allow long prompts for Perplexity-style goals

/**
 * Validates agent goal/prompt (optional field)
 * @param goal - The goal/prompt to validate
 * @returns Validation result
 */
export const validateGoal = (goal: string | undefined): ValidationResult => {
  if (goal === undefined || goal === null) {
    return { isValid: true, error: null }; // Optional field
  }

  if (typeof goal !== "string") {
    return {
      isValid: false,
      error: "El objetivo debe ser un texto",
    };
  }

  const trimmed = goal.trim();

  // Empty string is valid (field is optional)
  if (!trimmed) {
    return { isValid: true, error: null };
  }

  if (trimmed.length > MAX_GOAL_LENGTH) {
    return {
      isValid: false,
      error: `El objetivo no puede exceder ${MAX_GOAL_LENGTH} caracteres`,
    };
  }

  return { isValid: true, error: null };
};

export const VALIDATION_LIMITS = {
  MIN_NAME_LENGTH,
  MAX_NAME_LENGTH,
  MAX_DESCRIPTION_LENGTH,
  MAX_GOAL_LENGTH,
} as const;
