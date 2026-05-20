/**
 * Custom hook for sorting functionality.
 * Provides convenient sorting state and controls.
 */

import { useState, useMemo, useCallback } from 'react';
import { sortBy, sortByDesc } from '../utils/sorting';

/**
 * Sort configuration.
 */
export interface SortConfig<T> {
  key: keyof T | ((item: T) => number | string | Date);
  direction: 'asc' | 'desc';
}

/**
 * Options for useSort hook.
 */
export interface UseSortOptions<T> {
  items: T[];
  initialSort?: SortConfig<T>;
}

/**
 * Return type for useSort hook.
 */
export interface UseSortReturn<T> {
  sortedItems: T[];
  sortConfig: SortConfig<T> | null;
  sort: (key: keyof T | ((item: T) => number | string | Date), direction?: 'asc' | 'desc') => void;
  clearSort: () => void;
}

/**
 * Custom hook for sorting functionality.
 * Provides convenient sorting with state management.
 *
 * @param options - Hook options
 * @returns Sort state and controls
 */
export function useSort<T extends Record<string, any>>(
  options: UseSortOptions<T>
): UseSortReturn<T> {
  const { items, initialSort } = options;

  const [sortConfig, setSortConfig] = useState<SortConfig<T> | null>(
    initialSort || null
  );

  const sortedItems = useMemo(() => {
    if (!sortConfig) {
      return items;
    }

    const getValue =
      typeof sortConfig.key === 'function'
        ? sortConfig.key
        : (item: T) => item[sortConfig.key];

    return sortConfig.direction === 'asc'
      ? sortBy(items, getValue)
      : sortByDesc(items, getValue);
  }, [items, sortConfig]);

  const sort = useCallback(
    (
      key: keyof T | ((item: T) => number | string | Date),
      direction: 'asc' | 'desc' = 'asc'
    ) => {
      setSortConfig({ key, direction });
    },
    []
  );

  const clearSort = useCallback(() => {
    setSortConfig(null);
  }, []);

  return {
    sortedItems,
    sortConfig,
    sort,
    clearSort,
  };
}

