/**
 * Array utilities
 */

/**
 * Chunk array into smaller arrays
 */
export const chunk = <T,>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

/**
 * Remove duplicates from array
 */
export const unique = <T,>(array: T[]): T[] => {
  return Array.from(new Set(array));
};

/**
 * Remove duplicates by key
 */
export const uniqueBy = <T,>(array: T[], key: keyof T): T[] => {
  const seen = new Set();
  return array.filter((item) => {
    const value = item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
};

/**
 * Group array by key
 */
export const groupBy = <T,>(
  array: T[],
  key: keyof T | ((item: T) => string)
): Record<string, T[]> => {
  return array.reduce((groups, item) => {
    const groupKey =
      typeof key === 'function' ? key(item) : String(item[key]);
    if (!groups[groupKey]) {
      groups[groupKey] = [];
    }
    groups[groupKey].push(item);
    return groups;
  }, {} as Record<string, T[]>);
};

/**
 * Sort array by key
 */
export const sortBy = <T,>(
  array: T[],
  key: keyof T | ((item: T) => any),
  order: 'asc' | 'desc' = 'asc'
): T[] => {
  const sorted = [...array].sort((a, b) => {
    const aValue = typeof key === 'function' ? key(a) : a[key];
    const bValue = typeof key === 'function' ? key(b) : b[key];

    if (aValue < bValue) return order === 'asc' ? -1 : 1;
    if (aValue > bValue) return order === 'asc' ? 1 : -1;
    return 0;
  });

  return sorted;
};

/**
 * Shuffle array
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
 * Get random item from array
 */
export const randomItem = <T,>(array: T[]): T | undefined => {
  if (array.length === 0) return undefined;
  return array[Math.floor(Math.random() * array.length)];
};

/**
 * Get random items from array
 */
export const randomItems = <T,>(array: T[], count: number): T[] => {
  const shuffled = shuffle(array);
  return shuffled.slice(0, Math.min(count, array.length));
};

/**
 * Flatten nested arrays
 */
export const flatten = <T,>(array: (T | T[])[]): T[] => {
  return array.reduce((acc, item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : item);
  }, [] as T[]);
};

/**
 * Intersection of arrays
 */
export const intersection = <T,>(array1: T[], array2: T[]): T[] => {
  return array1.filter((item) => array2.includes(item));
};

/**
 * Difference of arrays
 */
export const difference = <T,>(array1: T[], array2: T[]): T[] => {
  return array1.filter((item) => !array2.includes(item));
};

/**
 * Union of arrays
 */
export const union = <T,>(array1: T[], array2: T[]): T[] => {
  return unique([...array1, ...array2]);
};

