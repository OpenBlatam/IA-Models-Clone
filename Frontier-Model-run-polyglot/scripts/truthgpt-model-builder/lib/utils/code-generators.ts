/**
 * Code Generator Utilities - Shared utilities for code generation
 */

/**
 * Convert string to PascalCase
 */
export function toPascalCase(str: string): string {
  return str
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('')
}

/**
 * Convert loss function name to PascalCase
 */
export function toPascalCaseLoss(loss: string): string {
  return loss
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('')
}

/**
 * Capitalize first letter
 */
export function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1)
}

/**
 * Format metrics for code
 */
export function formatMetricsForCode(metrics: string[]): string {
  return `['${metrics.join("', '")}']`
}
