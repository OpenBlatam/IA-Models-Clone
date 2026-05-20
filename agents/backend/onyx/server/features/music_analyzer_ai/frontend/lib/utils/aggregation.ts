/**
 * Aggregation utility functions.
 * Provides helper functions for aggregating data.
 */

/**
 * Calculates sum of array values.
 * @param items - Array of numbers
 * @param getValue - Function to get value (optional)
 * @returns Sum
 */
export function sum<T>(
  items: T[],
  getValue?: (item: T) => number
): number {
  if (getValue) {
    return items.reduce((acc, item) => acc + getValue(item), 0);
  }
  return (items as number[]).reduce((acc, val) => acc + val, 0);
}

/**
 * Calculates average of array values.
 * @param items - Array of numbers
 * @param getValue - Function to get value (optional)
 * @returns Average
 */
export function average<T>(
  items: T[],
  getValue?: (item: T) => number
): number {
  if (items.length === 0) return 0;
  return sum(items, getValue) / items.length;
}

/**
 * Gets minimum value from array.
 * @param items - Array of numbers
 * @param getValue - Function to get value (optional)
 * @returns Minimum value
 */
export function min<T>(
  items: T[],
  getValue?: (item: T) => number
): number | null {
  if (items.length === 0) return null;

  if (getValue) {
    return Math.min(...items.map(getValue));
  }
  return Math.min(...(items as number[]));
}

/**
 * Gets maximum value from array.
 * @param items - Array of numbers
 * @param getValue - Function to get value (optional)
 * @returns Maximum value
 */
export function max<T>(
  items: T[],
  getValue?: (item: T) => number
): number | null {
  if (items.length === 0) return null;

  if (getValue) {
    return Math.max(...items.map(getValue));
  }
  return Math.max(...(items as number[]));
}

/**
 * Counts items matching predicate.
 * @param items - Array to count
 * @param predicate - Count function (optional)
 * @returns Count
 */
export function count<T>(
  items: T[],
  predicate?: (item: T) => boolean
): number {
  if (predicate) {
    return items.filter(predicate).length;
  }
  return items.length;
}

/**
 * Calculates median of array values.
 * @param items - Array of numbers
 * @param getValue - Function to get value (optional)
 * @returns Median
 */
export function median<T>(
  items: T[],
  getValue?: (item: T) => number
): number | null {
  if (items.length === 0) return null;

  const values = getValue
    ? items.map(getValue).sort((a, b) => a - b)
    : [...(items as number[])].sort((a, b) => a - b);

  const mid = Math.floor(values.length / 2);

  if (values.length % 2 === 0) {
    return (values[mid - 1] + values[mid]) / 2;
  }

  return values[mid];
}

/**
 * Calculates mode (most frequent value) from array.
 * @param items - Array of values
 * @param getValue - Function to get value (optional)
 * @returns Mode value or null
 */
export function mode<T>(
  items: T[],
  getValue?: (item: T) => unknown
): T | null {
  if (items.length === 0) return null;

  const values = getValue ? items.map(getValue) : items;
  const frequency = new Map<unknown, number>();

  for (const value of values) {
    frequency.set(value, (frequency.get(value) || 0) + 1);
  }

  let maxFreq = 0;
  let modeValue: unknown = null;

  for (const [value, freq] of frequency.entries()) {
    if (freq > maxFreq) {
      maxFreq = freq;
      modeValue = value;
    }
  }

  if (getValue) {
    return items.find((item) => getValue(item) === modeValue) || null;
  }

  return (modeValue as T) || null;
}

