import { useState, useMemo, useCallback } from 'react';

export type SortDirection = 'asc' | 'desc';

export interface UseSortOptions<T> {
  initialSortKey?: keyof T;
  initialSortDirection?: SortDirection;
  sortFn?: (a: T, b: T, key: keyof T, direction: SortDirection) => number;
}

/**
 * Hook for sorting data
 */
export function useSort<T>(
  data: T[],
  options: UseSortOptions<T> = {}
): {
  sortKey: keyof T | null;
  sortDirection: SortDirection;
  sortedData: T[];
  setSortKey: (key: keyof T | null) => void;
  setSortDirection: (direction: SortDirection) => void;
  toggleSort: (key: keyof T) => void;
  clearSort: () => void;
} {
  const {
    initialSortKey,
    initialSortDirection = 'asc',
    sortFn,
  } = options;

  const [sortKey, setSortKey] = useState<keyof T | null>(initialSortKey ?? null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(initialSortDirection);

  const defaultSortFn = useCallback(
    (a: T, b: T, key: keyof T, direction: SortDirection): number => {
      const aValue = a[key];
      const bValue = b[key];

      if (aValue === bValue) return 0;

      let comparison = 0;
      if (aValue < bValue) {
        comparison = -1;
      } else if (aValue > bValue) {
        comparison = 1;
      }

      return direction === 'asc' ? comparison : -comparison;
    },
    []
  );

  const sortedData = useMemo(() => {
    if (!sortKey) return data;

    const sortFunction = sortFn || defaultSortFn;
    return [...data].sort((a, b) => sortFunction(a, b, sortKey, sortDirection));
  }, [data, sortKey, sortDirection, sortFn, defaultSortFn]);

  const toggleSort = useCallback((key: keyof T) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  }, [sortKey]);

  const clearSort = useCallback(() => {
    setSortKey(null);
    setSortDirection('asc');
  }, []);

  return {
    sortKey,
    sortDirection,
    sortedData,
    setSortKey,
    setSortDirection,
    toggleSort,
    clearSort,
  };
}



