import { useState, useCallback, useMemo } from 'react';

export type SortDirection = 'asc' | 'desc';

export type SortFunction<T> = (a: T, b: T) => number;

export interface SortConfig<T> {
  key?: keyof T;
  direction: SortDirection;
  compareFn?: SortFunction<T>;
}

export function useSort<T>(
  items: T[],
  initialConfig?: SortConfig<T>
) {
  const [sortConfig, setSortConfig] = useState<SortConfig<T> | null>(
    initialConfig || null
  );

  const sort = useCallback(
    (config: SortConfig<T>) => {
      setSortConfig(config);
    },
    []
  );

  const sortByKey = useCallback(
    (key: keyof T, direction: SortDirection = 'asc') => {
      setSortConfig({ key, direction });
    },
    []
  );

  const sortByFunction = useCallback(
    (compareFn: SortFunction<T>, direction: SortDirection = 'asc') => {
      setSortConfig({ compareFn, direction });
    },
    []
  );

  const toggleDirection = useCallback(() => {
    if (sortConfig) {
      setSortConfig({
        ...sortConfig,
        direction: sortConfig.direction === 'asc' ? 'desc' : 'asc',
      });
    }
  }, [sortConfig]);

  const clearSort = useCallback(() => {
    setSortConfig(null);
  }, []);

  const sortedItems = useMemo(() => {
    if (!sortConfig) return items;

    const sorted = [...items];

    sorted.sort((a, b) => {
      let comparison = 0;

      if (sortConfig.compareFn) {
        comparison = sortConfig.compareFn(a, b);
      } else if (sortConfig.key !== undefined) {
        const aValue = a[sortConfig.key];
        const bValue = b[sortConfig.key];

        if (aValue < bValue) comparison = -1;
        else if (aValue > bValue) comparison = 1;
      }

      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [items, sortConfig]);

  return {
    sortConfig,
    sortedItems,
    sort,
    sortByKey,
    sortByFunction,
    toggleDirection,
    clearSort,
    isSorted: sortConfig !== null,
  };
}


