import { useState, useCallback, useMemo } from 'react';

export type FilterFunction<T> = (item: T) => boolean;

export function useFilter<T>(
  items: T[],
  initialFilters: FilterFunction<T>[] = []
) {
  const [filters, setFilters] = useState<FilterFunction<T>[]>(initialFilters);

  const addFilter = useCallback((filter: FilterFunction<T>) => {
    setFilters((prev) => [...prev, filter]);
  }, []);

  const removeFilter = useCallback((filter: FilterFunction<T>) => {
    setFilters((prev) => prev.filter((f) => f !== filter));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters([]);
  }, []);

  const filteredItems = useMemo(() => {
    if (filters.length === 0) return items;

    return items.filter((item) => {
      return filters.every((filter) => filter(item));
    });
  }, [items, filters]);

  return {
    filters,
    filteredItems,
    addFilter,
    removeFilter,
    clearFilters,
    filterCount: filters.length,
    hasFilters: filters.length > 0,
  };
}


