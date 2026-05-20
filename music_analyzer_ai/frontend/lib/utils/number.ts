/**
 * Number utility functions.
 * Provides helper functions for number manipulation and formatting.
 */

/**
 * Clamps a number between min and max values.
 * @param value - Number to clamp
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clamped number
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Checks if a number is within a range.
 * @param value - Number to check
 * @param min - Minimum value (inclusive)
 * @param max - Maximum value (inclusive)
 * @returns True if number is in range
 */
export function isInRange(value: number, min: number, max: number): boolean {
  return value >= min && value <= max;
}

/**
 * Generates a random number between min and max (inclusive).
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Random number
 */
export function random(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Rounds a number to a specified number of decimal places.
 * @param value - Number to round
 * @param decimals - Number of decimal places (default: 0)
 * @returns Rounded number
 */
export function roundTo(value: number, decimals: number = 0): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

/**
 * Formats a number with thousand separators.
 * @param value - Number to format
 * @param locale - Locale string (default: 'es-ES')
 * @param options - Intl.NumberFormatOptions
 * @returns Formatted number string
 */
export function formatNumber(
  value: number,
  locale: string = 'es-ES',
  options?: Intl.NumberFormatOptions
): string {
  if (typeof value !== 'number' || isNaN(value)) {
    return '0';
  }

  return new Intl.NumberFormat(locale, options).format(value);
}

/**
 * Parses a string to a number with validation.
 * @param value - String to parse
 * @param defaultValue - Default value if parsing fails
 * @returns Parsed number or default
 */
export function parseNumber(
  value: string,
  defaultValue: number = 0
): number {
  if (typeof value !== 'string') {
    return defaultValue;
  }

  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

/**
 * Checks if a value is a valid number.
 * @param value - Value to check
 * @returns True if value is a valid number
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

/**
 * Converts bytes to human-readable format.
 * @param bytes - Bytes to convert
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Calculates percentage of a value.
 * @param value - Current value
 * @param total - Total value
 * @returns Percentage (0-100)
 */
export function calculatePercentage(value: number, total: number): number {
  if (total === 0) {
    return 0;
  }
  return clamp((value / total) * 100, 0, 100);
}

/**
 * Linearly interpolates between two numbers.
 * @param start - Start value
 * @param end - End value
 * @param t - Interpolation factor (0-1)
 * @returns Interpolated value
 */
export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * clamp(t, 0, 1);
}

