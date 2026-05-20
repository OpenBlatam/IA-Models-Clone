/**
 * Sorting utilities
 */

export type SortOrder = 'asc' | 'desc';

export interface SortConfig<T> {
  key: keyof T;
  order: SortOrder;
}

// Sort array by key
export function sortBy<T>(array: T[], key: keyof T, order: SortOrder = 'asc'): T[] {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];

    if (aVal === bVal) return 0;

    const comparison = aVal < bVal ? -1 : 1;
    return order === 'asc' ? comparison : -comparison;
  });
}

// Sort array by multiple keys
export function sortByMultiple<T>(array: T[], configs: SortConfig<T>[]): T[] {
  return [...array].sort((a, b) => {
    for (const config of configs) {
      const aVal = a[config.key];
      const bVal = b[config.key];

      if (aVal === bVal) continue;

      const comparison = aVal < bVal ? -1 : 1;
      const result = config.order === 'asc' ? comparison : -comparison;

      if (result !== 0) return result;
    }

    return 0;
  });
}

// Sort array by function
export function sortByFunction<T>(
  array: T[],
  fn: (item: T) => number | string,
  order: SortOrder = 'asc'
): T[] {
  return [...array].sort((a, b) => {
    const aVal = fn(a);
    const bVal = fn(b);

    if (aVal === bVal) return 0;

    const comparison = aVal < bVal ? -1 : 1;
    return order === 'asc' ? comparison : -comparison;
  });
}

// Natural sort (handles numbers in strings)
export function naturalSort(array: string[], order: SortOrder = 'asc'): string[] {
  return [...array].sort((a, b) => {
    const aNum = parseFloat(a);
    const bNum = parseFloat(b);

    if (!isNaN(aNum) && !isNaN(bNum)) {
      return order === 'asc' ? aNum - bNum : bNum - aNum;
    }

    return order === 'asc' ? a.localeCompare(b) : b.localeCompare(a);
  });
}



