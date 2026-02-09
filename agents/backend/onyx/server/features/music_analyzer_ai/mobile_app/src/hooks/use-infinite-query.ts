import { useState, useCallback, useEffect, useRef } from 'react';

interface UseInfiniteQueryOptions<T> {
  queryFn: (page: number) => Promise<{ data: T[]; hasMore: boolean }>;
  initialPage?: number;
  pageSize?: number;
  enabled?: boolean;
}

export function useInfiniteQuery<T>(options: UseInfiniteQueryOptions<T>) {
  const {
    queryFn,
    initialPage = 1,
    pageSize = 20,
    enabled = true,
  } = options;

  const [data, setData] = useState<T[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const isInitialMount = useRef(true);

  const fetchPage = useCallback(
    async (page: number, isLoadMore = false) => {
      if (isLoadMore) {
        setIsLoadingMore(true);
      } else {
        setIsLoading(true);
      }
      setError(null);

      try {
        const result = await queryFn(page);
        
        if (isLoadMore) {
          setData((prev) => [...prev, ...result.data]);
        } else {
          setData(result.data);
        }

        setHasMore(result.hasMore);
        setCurrentPage(page);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to fetch data'));
      } finally {
        setIsLoading(false);
        setIsLoadingMore(false);
      }
    },
    [queryFn]
  );

  useEffect(() => {
    if (enabled && isInitialMount.current) {
      isInitialMount.current = false;
      fetchPage(initialPage, false);
    }
  }, [enabled, initialPage, fetchPage]);

  const loadMore = useCallback(() => {
    if (hasMore && !isLoadingMore && !isLoading) {
      fetchPage(currentPage + 1, true);
    }
  }, [hasMore, isLoadingMore, isLoading, currentPage, fetchPage]);

  const refresh = useCallback(() => {
    setData([]);
    setCurrentPage(initialPage);
    setHasMore(true);
    fetchPage(initialPage, false);
  }, [initialPage, fetchPage]);

  const reset = useCallback(() => {
    setData([]);
    setCurrentPage(initialPage);
    setHasMore(true);
    setError(null);
    isInitialMount.current = true;
  }, [initialPage]);

  return {
    data,
    isLoading,
    isLoadingMore,
    error,
    hasMore,
    currentPage,
    loadMore,
    refresh,
    reset,
    refetch: refresh,
  };
}

