/**
 * Custom hook for pagination.
 * Provides convenient pagination state and controls.
 */

import { useState, useMemo, useCallback } from 'react';
import {
  calculatePagination,
  getPageItems,
  generatePageNumbers,
  type PaginationOptions,
  type PaginationResult,
} from '../utils/pagination';

/**
 * Options for usePagination hook.
 */
export interface UsePaginationOptions {
  total: number;
  initialPage?: number;
  initialPageSize?: number;
}

/**
 * Return type for usePagination hook.
 */
export interface UsePaginationReturn<T> {
  currentPage: number;
  pageSize: number;
  total: number;
  pagination: PaginationResult;
  pageItems: T[];
  pageNumbers: (number | 'ellipsis')[];
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  nextPage: () => void;
  previousPage: () => void;
  goToFirstPage: () => void;
  goToLastPage: () => void;
}

/**
 * Custom hook for pagination.
 * Provides convenient pagination with state management.
 *
 * @param items - Items to paginate
 * @param options - Pagination options
 * @returns Pagination state and controls
 */
export function usePagination<T>(
  items: T[],
  options: UsePaginationOptions
): UsePaginationReturn<T> {
  const {
    total,
    initialPage = 1,
    initialPageSize = 10,
  } = options;

  const [page, setPage] = useState(initialPage);
  const [pageSize, setPageSize] = useState(initialPageSize);

  const pagination = useMemo(
    () => calculatePagination({ page, pageSize, total }),
    [page, pageSize, total]
  );

  const pageItems = useMemo(
    () => getPageItems(items, page, pageSize),
    [items, page, pageSize]
  );

  const pageNumbers = useMemo(
    () => generatePageNumbers(pagination.currentPage, pagination.totalPages),
    [pagination.currentPage, pagination.totalPages]
  );

  const setPageSafe = useCallback(
    (newPage: number) => {
      setPage(Math.max(1, Math.min(newPage, pagination.totalPages)));
    },
    [pagination.totalPages]
  );

  const setPageSizeSafe = useCallback((newSize: number) => {
    setPageSize(Math.max(1, newSize));
    setPage(1); // Reset to first page when changing page size
  }, []);

  const nextPage = useCallback(() => {
    if (pagination.hasNextPage) {
      setPageSafe(pagination.nextPage!);
    }
  }, [pagination.hasNextPage, pagination.nextPage, setPageSafe]);

  const previousPage = useCallback(() => {
    if (pagination.hasPreviousPage) {
      setPageSafe(pagination.previousPage!);
    }
  }, [pagination.hasPreviousPage, pagination.previousPage, setPageSafe]);

  const goToFirstPage = useCallback(() => {
    setPageSafe(1);
  }, [setPageSafe]);

  const goToLastPage = useCallback(() => {
    setPageSafe(pagination.totalPages);
  }, [pagination.totalPages, setPageSafe]);

  return {
    currentPage: pagination.currentPage,
    pageSize,
    total,
    pagination,
    pageItems,
    pageNumbers,
    setPage: setPageSafe,
    setPageSize: setPageSizeSafe,
    nextPage,
    previousPage,
    goToFirstPage,
    goToLastPage,
  };
}

