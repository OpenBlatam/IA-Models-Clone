import { useMemo } from 'react';
import { useFilter } from './useFilter';
import { useSort } from './useSort';
import { usePagination } from './usePagination';

interface UseCombinedDataOptions<T> {
  data: T[];
  filterFn?: (item: T) => boolean;
  sortConfig?: { key: keyof T; direction: 'asc' | 'desc' };
  itemsPerPage?: number;
  initialPage?: number;
}

export const useCombinedData = <T,>({
  data,
  filterFn,
  sortConfig,
  itemsPerPage,
  initialPage = 1,
}: UseCombinedDataOptions<T>) => {
  const { filteredData, setFilter, clearFilter, isFiltered } = useFilter(
    data,
    filterFn
  );

  const { sortedData, sortConfig: currentSort, handleSort, clearSort } = useSort(
    filteredData,
    sortConfig
  );

  const {
    currentPage,
    totalPages,
    paginatedData,
    goToPage,
    nextPage,
    previousPage,
    goToFirstPage,
    goToLastPage,
    hasNextPage,
    hasPreviousPage,
  } = usePagination({
    data: sortedData,
    itemsPerPage,
    initialPage,
  });

  return {
    // Filter
    filteredData,
    setFilter,
    clearFilter,
    isFiltered,
    // Sort
    sortedData,
    sortConfig: currentSort,
    handleSort,
    clearSort,
    // Pagination
    currentPage,
    totalPages,
    paginatedData,
    goToPage,
    nextPage,
    previousPage,
    goToFirstPage,
    goToLastPage,
    hasNextPage,
    hasPreviousPage,
    // Combined
    processedData: paginatedData,
    totalItems: sortedData.length,
  };
};

