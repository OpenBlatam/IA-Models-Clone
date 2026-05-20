import { useState, useCallback } from 'react';
import { PAGINATION } from '../constants';

interface UsePaginationOptions {
  initialLimit?: number;
  initialOffset?: number;
}

interface UsePaginationReturn {
  limit: number;
  offset: number;
  currentPage: number;
  goToPage: (page: number) => void;
  nextPage: () => void;
  previousPage: () => void;
  reset: () => void;
  setLimit: (limit: number) => void;
}

export const usePagination = ({
  initialLimit = PAGINATION.DEFAULT_LIMIT,
  initialOffset = PAGINATION.DEFAULT_OFFSET,
}: UsePaginationOptions = {}): UsePaginationReturn => {
  const [limit, setLimit] = useState(initialLimit);
  const [offset, setOffset] = useState(initialOffset);

  const currentPage = Math.floor(offset / limit) + 1;

  const goToPage = useCallback((page: number): void => {
    setOffset((page - 1) * limit);
  }, [limit]);

  const nextPage = useCallback((): void => {
    setOffset((prev) => prev + limit);
  }, [limit]);

  const previousPage = useCallback((): void => {
    setOffset((prev) => Math.max(0, prev - limit));
  }, [limit]);

  const reset = useCallback((): void => {
    setOffset(PAGINATION.DEFAULT_OFFSET);
  }, []);

  return {
    limit,
    offset,
    currentPage,
    goToPage,
    nextPage,
    previousPage,
    reset,
    setLimit,
  };
};

