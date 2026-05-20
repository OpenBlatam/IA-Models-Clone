/**
 * Comparison utility functions.
 * Provides helper functions for comparing values.
 */

/**
 * Compares two values for sorting.
 * @param a - First value
 * @param b - Second value
 * @returns Comparison result (-1, 0, 1)
 */
export function compare(a: any, b: any): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/**
 * Compares two values (case-insensitive strings).
 * @param a - First value
 * @param b - Second value
 * @returns Comparison result (-1, 0, 1)
 */
export function compareIgnoreCase(a: string, b: string): number {
  return a.toLowerCase().localeCompare(b.toLowerCase());
}

/**
 * Compares two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Comparison result (-1, 0, 1)
 */
export function compareNumbers(a: number, b: number): number {
  return a - b;
}

/**
 * Compares two dates.
 * @param a - First date
 * @param b - Second date
 * @returns Comparison result (-1, 0, 1)
 */
export function compareDates(
  a: Date | string | number,
  b: Date | string | number
): number {
  const dateA = new Date(a).getTime();
  const dateB = new Date(b).getTime();
  return dateA - dateB;
}

/**
 * Checks if two values are equal (deep comparison).
 * @param a - First value
 * @param b - Second value
 * @returns True if values are equal
 */
export function isEqual(a: any, b: any): boolean {
  if (a === b) return true;

  if (a == null || b == null) return false;

  if (typeof a !== typeof b) return false;

  if (typeof a !== 'object') return a === b;

  if (Array.isArray(a) !== Array.isArray(b)) return false;

  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!isEqual(a[i], b[i])) return false;
    }
    return true;
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!keysB.includes(key)) return false;
    if (!isEqual(a[key], b[key])) return false;
  }

  return true;
}

/**
 * Checks if value is greater than other.
 * @param a - First value
 * @param b - Second value
 * @returns True if a > b
 */
export function isGreaterThan(a: number, b: number): boolean {
  return a > b;
}

/**
 * Checks if value is less than other.
 * @param a - First value
 * @param b - Second value
 * @returns True if a < b
 */
export function isLessThan(a: number, b: number): boolean {
  return a < b;
}

/**
 * Checks if value is between two values (inclusive).
 * @param value - Value to check
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns True if value is between min and max
 */
export function isBetween(value: number, min: number, max: number): boolean {
  return value >= min && value <= max;
}

