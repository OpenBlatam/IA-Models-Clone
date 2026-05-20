/**
 * Formatting utilities for numbers, credits, and other data types
 */
const CREDITS_LOW_THRESHOLD = 100;
const CREDITS_WARNING_THRESHOLD = 1000;

export const formatNumber = (value: number | null | undefined): string => {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return value.toLocaleString("es-ES");
};

export const formatCredits = (credits: number | null | undefined): string => {
  return formatNumber(credits);
};

export const getCreditsStatusClass = (
  credits: number | null | undefined
): string => {
  if (credits === null || credits === undefined) {
    return "";
  }

  if (credits < CREDITS_LOW_THRESHOLD) {
    return "text-red-600 font-semibold";
  }

  if (credits < CREDITS_WARNING_THRESHOLD) {
    return "text-yellow-600 font-semibold";
  }

  return "";
};

export const getCreditsStatusAriaLabel = (
  credits: number | null | undefined
): string => {
  if (credits === null || credits === undefined) {
    return "Créditos no disponibles";
  }

  if (credits < CREDITS_LOW_THRESHOLD) {
    return `Créditos bajos: ${formatCredits(credits)}`;
  }

  if (credits < CREDITS_WARNING_THRESHOLD) {
    return `Créditos limitados: ${formatCredits(credits)}`;
  }

  return `Créditos disponibles: ${formatCredits(credits)}`;
};







