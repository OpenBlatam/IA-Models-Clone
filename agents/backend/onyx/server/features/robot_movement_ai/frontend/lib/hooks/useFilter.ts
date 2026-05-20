import { useState, useMemo, useCallback } from 'react';

export interface UseFilterOptions<T> {
  initialFilter?: string;
  filterFn?: (item: T, filter: string) => boolean;
  searchFields?: Array<keyof T | ((item: T) => string)>;
}

/**
 * Hook for filtering data
 */
export function useFilter<T>(
  data: T[],
  options: UseFilterOptions<T> = {}
): {
  filter: string;
  setFilter: (filter: string) => void;
  filteredData: T[];
  clearFilter: () => void;
} {
  const { initialFilter = '', filterFn, searchFields } = options;
  const [filter, setFilter] = useState(initialFilter);

  const defaultFilterFn = useCallback(
    (item: T, filterValue: string): boolean => {
      if (!filterValue) return true;

      const lowerFilter = filterValue.toLowerCase();

      if (searchFields) {
        return searchFields.some((field) => {
          if (typeof field === 'function') {
            const value = field(item);
            return String(value).toLowerCase().includes(lowerFilter);
          }
          const value = item[field];
          return String(value).toLowerCase().includes(lowerFilter);
        });
      }

      // Default: search in all string properties
      return Object.values(item as any).some((value) =>
        String(value).toLowerCase().includes(lowerFilter)
      );
    },
    [searchFields]
  );

  const filteredData = useMemo(() => {
    const filterFunction = filterFn || defaultFilterFn;
    return data.filter((item) => filterFunction(item, filter));
  }, [data, filter, filterFn, defaultFilterFn]);

  const clearFilter = useCallback(() => {
    setFilter('');
  }, []);

  return {
    filter,
    setFilter,
    filteredData,
    clearFilter,
  };
}



