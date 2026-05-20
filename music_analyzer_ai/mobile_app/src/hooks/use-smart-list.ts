import { useState, useCallback, useMemo } from 'react';
import type { ListRenderItem } from 'react-native';

interface UseSmartListOptions<T> {
  initialData?: T[];
  onLoadMore?: () => Promise<void>;
  hasMore?: boolean;
  pageSize?: number;
}

export function useSmartList<T>(options: UseSmartListOptions<T> = {}) {
  const { initialData = [], onLoadMore, hasMore = false, pageSize = 20 } = options;
  
  const [data, setData] = useState<T[]>(initialData);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadMore = useCallback(async () => {
    if (!onLoadMore || isLoadingMore || !hasMore) {
      return;
    }

    setIsLoadingMore(true);
    setError(null);

    try {
      await onLoadMore();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load more'));
    } finally {
      setIsLoadingMore(false);
    }
  }, [onLoadMore, isLoadingMore, hasMore]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setError(null);

    try {
      if (onLoadMore) {
        await onLoadMore();
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to refresh'));
    } finally {
      setRefreshing(false);
    }
  }, [onLoadMore]);

  const updateData = useCallback((newData: T[]) => {
    setData(newData);
  }, []);

  const appendData = useCallback((newItems: T[]) => {
    setData((prev) => [...prev, ...newItems]);
  }, []);

  const prependData = useCallback((newItems: T[]) => {
    setData((prev) => [...newItems, ...prev]);
  }, []);

  const removeItem = useCallback((predicate: (item: T) => boolean) => {
    setData((prev) => prev.filter((item) => !predicate(item)));
  }, []);

  const updateItem = useCallback((predicate: (item: T) => boolean, updater: (item: T) => T) => {
    setData((prev) => prev.map((item) => (predicate(item) ? updater(item) : item)));
  }, []);

  const clear = useCallback(() => {
    setData([]);
    setError(null);
  }, []);

  const retry = useCallback(() => {
    setError(null);
    if (onLoadMore) {
      loadMore();
    }
  }, [onLoadMore, loadMore]);

  return {
    data,
    isLoading,
    isLoadingMore,
    error,
    refreshing,
    hasMore,
    loadMore,
    refresh,
    updateData,
    appendData,
    prependData,
    removeItem,
    updateItem,
    clear,
    retry,
  };
}

