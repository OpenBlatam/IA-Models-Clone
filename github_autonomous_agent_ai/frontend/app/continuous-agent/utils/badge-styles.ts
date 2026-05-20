import { BADGE_STYLES } from "../constants/styles";
import type { PromptValidationResult } from "./prompt-validation";

/**
 * Gets badge style class based on validation result
 * @param validation - Validation result
 * @returns Badge style class name
 */
export const getValidationBadgeStyle = (
  validation: PromptValidationResult
): string => {
  const { isValid, errors } = validation;
  if (isValid && errors.length === 0) {
    return BADGE_STYLES.SUCCESS;
  }
  if (errors.length > 0) {
    return BADGE_STYLES.ERROR;
  }
  return BADGE_STYLES.WARNING;
};

/**
 * Gets validation badge title/tooltip text
 * @param validation - Validation result
 * @returns Tooltip text
 */
export const getValidationBadgeTitle = (
  validation: PromptValidationResult
): string => {
  const { isValid, errors, warnings } = validation;
  if (isValid && errors.length === 0) {
    return "Prompt válido";
  }
  if (errors.length > 0) {
    return `Errores: ${errors.join(", ")}`;
  }
  return `Advertencias: ${warnings.join(", ")}`;
};

/**
 * Gets validation badge status text
 * @param validation - Validation result
 * @returns Status text
 */
export const getValidationBadgeText = (
  validation: PromptValidationResult
): string => {
  const { isValid, errors, warnings } = validation;
  if (isValid && errors.length === 0) {
    return "Válido";
  }
  if (errors.length > 0) {
    return `${errors.length} error${errors.length !== 1 ? "es" : ""}`;
  }
  return `${warnings.length} advertencia${warnings.length !== 1 ? "s" : ""}`;
};

/**
 * Gets validation badge icon
 * @param validation - Validation result
 * @returns Icon character
 */
export const getValidationBadgeIcon = (
  validation: PromptValidationResult
): string => {
  const { isValid, errors } = validation;
  if (isValid && errors.length === 0) {
    return "✓";
  }
  if (errors.length > 0) {
    return "✗";
  }
  return "⚠";
};



