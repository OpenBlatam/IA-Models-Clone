/**
 * Sorting utilities
 */

export type SortDirection = 'asc' | 'desc';

/**
 * Sort array by key
 */
export const sortBy = <T,>(
  array: T[],
  key: keyof T,
  direction: SortDirection = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aValue = a[key];
    const bValue = b[key];

    if (aValue === null || aValue === undefined) return 1;
    if (bValue === null || bValue === undefined) return -1;

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      const comparison = aValue.localeCompare(bValue);
      return direction === 'asc' ? comparison : -comparison;
    }

    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return direction === 'asc' ? aValue - bValue : bValue - aValue;
    }

    if (aValue instanceof Date && bValue instanceof Date) {
      return direction === 'asc'
        ? aValue.getTime() - bValue.getTime()
        : bValue.getTime() - aValue.getTime();
    }

    return 0;
  });
};

/**
 * Sort array by multiple keys
 */
export const sortByMultiple = <T,>(
  array: T[],
  keys: Array<{ key: keyof T; direction: SortDirection }>
): T[] => {
  return [...array].sort((a, b) => {
    for (const { key, direction } of keys) {
      const aValue = a[key];
      const bValue = b[key];

      if (aValue === null || aValue === undefined) {
        if (bValue === null || bValue === undefined) continue;
        return 1;
      }
      if (bValue === null || bValue === undefined) return -1;

      let comparison = 0;

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        comparison = aValue.localeCompare(bValue);
      } else if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue;
      } else if (aValue instanceof Date && bValue instanceof Date) {
        comparison = aValue.getTime() - bValue.getTime();
      }

      if (comparison !== 0) {
        return direction === 'asc' ? comparison : -comparison;
      }
    }
    return 0;
  });
};

/**
 * Sort array by custom comparator
 */
export const sortWith = <T,>(
  array: T[],
  comparator: (a: T, b: T) => number
): T[] => {
  return [...array].sort(comparator);
};

