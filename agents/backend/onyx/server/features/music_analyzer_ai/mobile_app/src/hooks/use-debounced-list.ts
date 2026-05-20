import { useState, useEffect, useCallback, useMemo } from 'react';
import { useDebounce } from './use-debounce';

interface UseDebouncedListOptions<T> {
  items: T[];
  searchQuery?: string;
  searchFn?: (item: T, query: string) => boolean;
  filterFn?: (item: T) => boolean;
  sortFn?: (a: T, b: T) => number;
  debounceMs?: number;
}

export function useDebouncedList<T>(options: UseDebouncedListOptions<T>) {
  const {
    items,
    searchQuery = '',
    searchFn,
    filterFn,
    sortFn,
    debounceMs = 300,
  } = options;

  const [filteredItems, setFilteredItems] = useState<T[]>(items);
  const debouncedQuery = useDebounce(searchQuery, debounceMs);

  useEffect(() => {
    let result = [...items];

    // Apply search filter
    if (debouncedQuery && searchFn) {
      const query = debouncedQuery.toLowerCase().trim();
      result = result.filter((item) => searchFn(item, query));
    }

    // Apply custom filter
    if (filterFn) {
      result = result.filter(filterFn);
    }

    // Apply sorting
    if (sortFn) {
      result = result.sort(sortFn);
    }

    setFilteredItems(result);
  }, [items, debouncedQuery, searchFn, filterFn, sortFn]);

  const stats = useMemo(
    () => ({
      total: items.length,
      filtered: filteredItems.length,
      isFiltered: filteredItems.length !== items.length,
    }),
    [items.length, filteredItems.length]
  );

  return {
    filteredItems,
    stats,
    query: debouncedQuery,
  };
}

