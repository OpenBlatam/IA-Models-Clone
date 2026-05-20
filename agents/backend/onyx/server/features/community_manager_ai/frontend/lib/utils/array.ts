/**
 * Array Utility Functions
 * Utility functions for array manipulation
 */

/**
 * Removes duplicate values from an array
 * @param array - Array to deduplicate
 * @returns New array with unique values
 */
export const unique = <T,>(array: T[]): T[] => {
  return Array.from(new Set(array));
};

/**
 * Groups array items by a key function
 * @param array - Array to group
 * @param keyFn - Function to extract the key
 * @returns Object with grouped items
 */
export const groupBy = <T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> => {
  return array.reduce((acc, item) => {
    const key = keyFn(item);
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  }, {} as Record<K, T[]>);
};

/**
 * Chunks an array into smaller arrays of specified size
 * @param array - Array to chunk
 * @param size - Size of each chunk
 * @returns Array of chunks
 */
export const chunk = <T,>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

/**
 * Flattens a nested array
 * @param array - Nested array to flatten
 * @param depth - Flattening depth (default: Infinity)
 * @returns Flattened array
 */
export const flatten = <T,>(array: (T | T[])[], depth: number = Infinity): T[] => {
  return array.flat(depth) as T[];
};

/**
 * Gets the difference between two arrays
 * @param array1 - First array
 * @param array2 - Second array
 * @returns Array of items in array1 but not in array2
 */
export const difference = <T,>(array1: T[], array2: T[]): T[] => {
  const set2 = new Set(array2);
  return array1.filter((item) => !set2.has(item));
};

/**
 * Gets the intersection of two arrays
 * @param array1 - First array
 * @param array2 - Second array
 * @returns Array of items in both arrays
 */
export const intersection = <T,>(array1: T[], array2: T[]): T[] => {
  const set2 = new Set(array2);
  return array1.filter((item) => set2.has(item));
};

/**
 * Shuffles an array (Fisher-Yates algorithm)
 * @param array - Array to shuffle
 * @returns New shuffled array
 */
export const shuffle = <T,>(array: T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

/**
 * Gets a random item from an array
 * @param array - Array to sample from
 * @returns Random item or undefined if array is empty
 */
export const sample = <T,>(array: T[]): T | undefined => {
  if (array.length === 0) return undefined;
  return array[Math.floor(Math.random() * array.length)];
};

/**
 * Gets the last item from an array
 * @param array - Array to get last item from
 * @returns Last item or undefined if array is empty
 */
export const last = <T,>(array: T[]): T | undefined => {
  return array[array.length - 1];
};

/**
 * Gets the first item from an array
 * @param array - Array to get first item from
 * @returns First item or undefined if array is empty
 */
export const first = <T,>(array: T[]): T | undefined => {
  return array[0];
};


