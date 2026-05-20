/**
 * Sorting utility functions.
 * Provides helper functions for sorting arrays.
 */

/**
 * Sorts array by property in ascending order.
 * @param items - Array to sort
 * @param getValue - Function to get sort value
 * @returns Sorted array
 */
export function sortBy<T>(
  items: T[],
  getValue: (item: T) => number | string | Date
): T[] {
  return [...items].sort((a, b) => {
    const aVal = getValue(a);
    const bVal = getValue(b);

    if (aVal < bVal) return -1;
    if (aVal > bVal) return 1;
    return 0;
  });
}

/**
 * Sorts array by property in descending order.
 * @param items - Array to sort
 * @param getValue - Function to get sort value
 * @returns Sorted array
 */
export function sortByDesc<T>(
  items: T[],
  getValue: (item: T) => number | string | Date
): T[] {
  return [...items].sort((a, b) => {
    const aVal = getValue(a);
    const bVal = getValue(b);

    if (aVal > bVal) return -1;
    if (aVal < bVal) return 1;
    return 0;
  });
}

/**
 * Sorts array by multiple properties.
 * @param items - Array to sort
 * @param sorters - Array of sort functions
 * @returns Sorted array
 */
export function sortByMultiple<T>(
  items: T[],
  sorters: Array<(item: T) => number | string | Date>
): T[] {
  return [...items].sort((a, b) => {
    for (const getValue of sorters) {
      const aVal = getValue(a);
      const bVal = getValue(b);

      if (aVal < bVal) return -1;
      if (aVal > bVal) return 1;
    }
    return 0;
  });
}

/**
 * Sorts array with custom comparator.
 * @param items - Array to sort
 * @param compareFn - Comparison function
 * @returns Sorted array
 */
export function sortWith<T>(
  items: T[],
  compareFn: (a: T, b: T) => number
): T[] {
  return [...items].sort(compareFn);
}

/**
 * Reverses array order.
 * @param items - Array to reverse
 * @returns Reversed array
 */
export function reverse<T>(items: T[]): T[] {
  return [...items].reverse();
}

/**
 * Shuffles array (Fisher-Yates algorithm).
 * @param items - Array to shuffle
 * @returns Shuffled array
 */
export function shuffle<T>(items: T[]): T[] {
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

