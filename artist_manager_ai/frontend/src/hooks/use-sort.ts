import { useState, useMemo } from 'react';

type SortFunction<T> = (a: T, b: T) => number;
type SortKey<T> = keyof T | ((item: T) => string | number);

interface UseSortOptions<T> {
  items: T[];
  initialSortKey?: SortKey<T>;
  initialSortDirection?: 'asc' | 'desc';
}

interface SortResult<T> {
  sortedItems: T[];
  sortKey: SortKey<T> | null;
  sortDirection: 'asc' | 'desc';
  setSortKey: (key: SortKey<T>) => void;
  toggleSortDirection: () => void;
  clearSort: () => void;
  isSorted: boolean;
}

export const useSort = <T,>({
  items,
  initialSortKey,
  initialSortDirection = 'asc',
}: UseSortOptions<T>): SortResult<T> => {
  const [sortKey, setSortKeyState] = useState<SortKey<T> | null>(initialSortKey || null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>(initialSortDirection);

  const sortedItems = useMemo(() => {
    if (!sortKey) {
      return items;
    }

    const sorted = [...items].sort((a, b) => {
      const getValue = (item: T): string | number => {
        if (typeof sortKey === 'function') {
          return sortKey(item);
        }
        return item[sortKey] as string | number;
      };

      const aValue = getValue(a);
      const bValue = getValue(b);

      if (aValue < bValue) {
        return sortDirection === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortDirection === 'asc' ? 1 : -1;
      }
      return 0;
    });

    return sorted;
  }, [items, sortKey, sortDirection]);

  const setSortKey = (key: SortKey<T>) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKeyState(key);
      setSortDirection('asc');
    }
  };

  const toggleSortDirection = () => {
    setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
  };

  const clearSort = () => {
    setSortKeyState(null);
    setSortDirection('asc');
  };

  const isSorted = sortKey !== null;

  return {
    sortedItems,
    sortKey,
    sortDirection,
    setSortKey,
    toggleSortDirection,
    clearSort,
    isSorted,
  };
};

