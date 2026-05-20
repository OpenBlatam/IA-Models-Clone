import { useState, useMemo } from 'react';

type FilterFunction<T> = (item: T) => boolean;

interface UseFilterOptions<T> {
  items: T[];
  initialFilter?: FilterFunction<T>;
}

interface FilterResult<T> {
  filteredItems: T[];
  setFilter: (filter: FilterFunction<T> | null) => void;
  clearFilter: () => void;
  isFiltered: boolean;
}

export const useFilter = <T,>({
  items,
  initialFilter,
}: UseFilterOptions<T>): FilterResult<T> => {
  const [filter, setFilterState] = useState<FilterFunction<T> | null>(initialFilter || null);

  const filteredItems = useMemo(() => {
    if (!filter) {
      return items;
    }
    return items.filter(filter);
  }, [items, filter]);

  const setFilter = (newFilter: FilterFunction<T> | null) => {
    setFilterState(() => newFilter);
  };

  const clearFilter = () => {
    setFilterState(null);
  };

  const isFiltered = filter !== null;

  return {
    filteredItems,
    setFilter,
    clearFilter,
    isFiltered,
  };
};

