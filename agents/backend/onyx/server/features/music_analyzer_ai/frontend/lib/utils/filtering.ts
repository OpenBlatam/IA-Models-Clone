/**
 * Filtering utility functions.
 * Provides helper functions for filtering arrays.
 */

/**
 * Filters array by predicate function.
 * @param items - Array to filter
 * @param predicate - Filter function
 * @returns Filtered array
 */
export function filter<T>(
  items: T[],
  predicate: (item: T, index: number) => boolean
): T[] {
  return items.filter(predicate);
}

/**
 * Filters array to remove falsy values.
 * @param items - Array to filter
 * @returns Filtered array
 */
export function compact<T>(items: (T | null | undefined | false | 0 | '')[]): T[] {
  return items.filter((item): item is T => Boolean(item)) as T[];
}

/**
 * Filters array to unique items.
 * @param items - Array to filter
 * @param getKey - Function to get unique key
 * @returns Filtered array
 */
export function uniqueBy<T>(
  items: T[],
  getKey: (item: T) => unknown
): T[] {
  const seen = new Set<unknown>();
  return items.filter((item) => {
    const key = getKey(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

/**
 * Filters array by multiple predicates (AND).
 * @param items - Array to filter
 * @param predicates - Array of filter functions
 * @returns Filtered array
 */
export function filterByAll<T>(
  items: T[],
  predicates: Array<(item: T) => boolean>
): T[] {
  return items.filter((item) => predicates.every((pred) => pred(item)));
}

/**
 * Filters array by multiple predicates (OR).
 * @param items - Array to filter
 * @param predicates - Array of filter functions
 * @returns Filtered array
 */
export function filterByAny<T>(
  items: T[],
  predicates: Array<(item: T) => boolean>
): T[] {
  return items.filter((item) => predicates.some((pred) => pred(item)));
}

/**
 * Filters array by property value.
 * @param items - Array to filter
 * @param getValue - Function to get property value
 * @param value - Value to match
 * @returns Filtered array
 */
export function filterByValue<T>(
  items: T[],
  getValue: (item: T) => unknown,
  value: unknown
): T[] {
  return items.filter((item) => getValue(item) === value);
}

/**
 * Filters array by property range.
 * @param items - Array to filter
 * @param getValue - Function to get property value
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Filtered array
 */
export function filterByRange<T>(
  items: T[],
  getValue: (item: T) => number,
  min: number,
  max: number
): T[] {
  return items.filter((item) => {
    const value = getValue(item);
    return value >= min && value <= max;
  });
}

