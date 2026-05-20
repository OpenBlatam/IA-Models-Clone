import { useMemo } from 'react';
import { usePagination, UsePaginationOptions } from './usePagination';
import { useFilter, UseFilterOptions } from './useFilter';
import { useSort, UseSortOptions } from './useSort';

export interface UseTableOptions<T> extends UsePaginationOptions, UseFilterOptions<T>, UseSortOptions<T> {}

/**
 * Combined hook for table functionality (pagination, filtering, sorting)
 */
export function useTable<T>(
  data: T[],
  options: UseTableOptions<T> = {}
) {
  const { initialPage, initialPageSize, totalItems, ...filterOptions } = options;
  const { initialSortKey, initialSortDirection, sortFn, ...restFilterOptions } = filterOptions;
  const { initialFilter, filterFn, searchFields } = restFilterOptions;

  // Apply filter first
  const { filter, setFilter, filteredData, clearFilter } = useFilter(data, {
    initialFilter,
    filterFn,
    searchFields,
  });

  // Then apply sort
  const {
    sortKey,
    sortDirection,
    sortedData,
    setSortKey,
    setSortDirection,
    toggleSort,
    clearSort,
  } = useSort(filteredData, {
    initialSortKey,
    initialSortDirection,
    sortFn,
  });

  // Finally apply pagination
  const pagination = usePagination(sortedData, {
    initialPage,
    initialPageSize,
    totalItems,
  });

  return {
    // Filter
    filter,
    setFilter,
    clearFilter,
    // Sort
    sortKey,
    sortDirection,
    setSortKey,
    setSortDirection,
    toggleSort,
    clearSort,
    // Pagination
    ...pagination,
    // Combined
    data: pagination.paginatedData,
    originalData: data,
  };
}



