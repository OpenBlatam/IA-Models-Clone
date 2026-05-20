import { useMemo } from 'react';

interface QueryState<T> {
  data?: T;
  isLoading: boolean;
  error: Error | null;
}

interface UseQueryStateOptions {
  showLoading?: boolean;
  showError?: boolean;
  showEmpty?: boolean;
}

interface UseQueryStateReturn<T> {
  isLoading: boolean;
  error: Error | null;
  data: T | undefined;
  isEmpty: boolean;
  shouldShowLoading: boolean;
  shouldShowError: boolean;
  shouldShowEmpty: boolean;
  shouldShowContent: boolean;
}

export const useQueryState = <T,>(
  query: QueryState<T>,
  options: UseQueryStateOptions = {}
): UseQueryStateReturn<T> => {
  const {
    showLoading = true,
    showError = true,
    showEmpty = true,
  } = options;

  const isEmpty = useMemo(() => {
    if (!query.data) return true;
    if (Array.isArray(query.data)) return query.data.length === 0;
    return false;
  }, [query.data]);

  const shouldShowLoading = showLoading && query.isLoading;
  const shouldShowError = showError && !!query.error;
  const shouldShowEmpty = showEmpty && !query.isLoading && !query.error && isEmpty;
  const shouldShowContent = !shouldShowLoading && !shouldShowError && !shouldShowEmpty;

  return {
    isLoading: query.isLoading,
    error: query.error,
    data: query.data,
    isEmpty,
    shouldShowLoading,
    shouldShowError,
    shouldShowEmpty,
    shouldShowContent,
  };
};

