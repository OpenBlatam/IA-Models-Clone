import { useMemo } from 'react';

interface QueryState {
  isLoading: boolean;
  error: Error | null;
}

interface UseMultipleQueriesOptions {
  showLoading?: boolean;
  showError?: boolean;
}

interface UseMultipleQueriesReturn {
  isLoading: boolean;
  error: Error | null;
  hasError: boolean;
  shouldShowLoading: boolean;
  shouldShowError: boolean;
  shouldShowContent: boolean;
}

export const useMultipleQueries = (
  queries: QueryState[],
  options: UseMultipleQueriesOptions = {}
): UseMultipleQueriesReturn => {
  const {
    showLoading = true,
    showError = true,
  } = options;

  const isLoading = useMemo(
    () => queries.some((query) => query.isLoading),
    [queries]
  );

  const error = useMemo(
    () => queries.find((query) => query.error)?.error || null,
    [queries]
  );

  const hasError = !!error;
  const shouldShowLoading = showLoading && isLoading;
  const shouldShowError = showError && hasError;
  const shouldShowContent = !shouldShowLoading && !shouldShowError;

  return {
    isLoading,
    error,
    hasError,
    shouldShowLoading,
    shouldShowError,
    shouldShowContent,
  };
};

