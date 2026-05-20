/**
 * Array utility functions.
 * Provides helper functions for common array operations.
 */

/**
 * Removes duplicate items from an array.
 * @param array - Array to deduplicate
 * @param keyFn - Optional function to extract key for comparison
 * @returns Array with duplicates removed
 */
export function unique<T>(
  array: T[],
  keyFn?: (item: T) => unknown
): T[] {
  if (!Array.isArray(array)) {
    return [];
  }

  if (keyFn) {
    const seen = new Set<unknown>();
    return array.filter((item) => {
      const key = keyFn(item);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  return Array.from(new Set(array));
}

/**
 * Groups array items by a key function.
 * @param array - Array to group
 * @param keyFn - Function to extract key for grouping
 * @returns Object with grouped items
 */
export function groupBy<T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> {
  if (!Array.isArray(array)) {
    return {} as Record<K, T[]>;
  }

  return array.reduce(
    (acc, item) => {
      const key = keyFn(item);
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(item);
      return acc;
    },
    {} as Record<K, T[]>
  );
}

/**
 * Chunks an array into smaller arrays of specified size.
 * @param array - Array to chunk
 * @param size - Size of each chunk
 * @returns Array of chunks
 */
export function chunk<T>(array: T[], size: number): T[][] {
  if (!Array.isArray(array) || size <= 0) {
    return [];
  }

  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

/**
 * Flattens a nested array.
 * @param array - Nested array to flatten
 * @param depth - Depth to flatten (default: Infinity)
 * @returns Flattened array
 */
export function flatten<T>(array: unknown[], depth: number = Infinity): T[] {
  if (!Array.isArray(array)) {
    return [];
  }

  return array.flat(depth) as T[];
}

/**
 * Sorts an array by a key function.
 * @param array - Array to sort
 * @param keyFn - Function to extract sort key
 * @param order - Sort order ('asc' | 'desc')
 * @returns Sorted array
 */
export function sortBy<T>(
  array: T[],
  keyFn: (item: T) => number | string,
  order: 'asc' | 'desc' = 'asc'
): T[] {
  if (!Array.isArray(array)) {
    return [];
  }

  const sorted = [...array].sort((a, b) => {
    const aKey = keyFn(a);
    const bKey = keyFn(b);

    if (aKey < bKey) return order === 'asc' ? -1 : 1;
    if (aKey > bKey) return order === 'asc' ? 1 : -1;
    return 0;
  });

  return sorted;
}

/**
 * Partitions an array into two arrays based on a predicate.
 * @param array - Array to partition
 * @param predicate - Function to test each item
 * @returns Tuple of [matching items, non-matching items]
 */
export function partition<T>(
  array: T[],
  predicate: (item: T) => boolean
): [T[], T[]] {
  if (!Array.isArray(array)) {
    return [[], []];
  }

  const matching: T[] = [];
  const nonMatching: T[] = [];

  array.forEach((item) => {
    if (predicate(item)) {
      matching.push(item);
    } else {
      nonMatching.push(item);
    }
  });

  return [matching, nonMatching];
}

/**
 * Gets the intersection of two arrays.
 * @param array1 - First array
 * @param array2 - Second array
 * @returns Array of common items
 */
export function intersection<T>(array1: T[], array2: T[]): T[] {
  if (!Array.isArray(array1) || !Array.isArray(array2)) {
    return [];
  }

  const set2 = new Set(array2);
  return array1.filter((item) => set2.has(item));
}

/**
 * Gets the difference of two arrays (items in array1 but not in array2).
 * @param array1 - First array
 * @param array2 - Second array
 * @returns Array of items in array1 but not in array2
 */
export function difference<T>(array1: T[], array2: T[]): T[] {
  if (!Array.isArray(array1) || !Array.isArray(array2)) {
    return array1 || [];
  }

  const set2 = new Set(array2);
  return array1.filter((item) => !set2.has(item));
}

