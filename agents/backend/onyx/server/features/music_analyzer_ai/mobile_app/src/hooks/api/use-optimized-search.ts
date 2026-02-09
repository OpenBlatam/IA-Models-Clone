import { useState, useMemo } from 'react';
import { useDebounce } from '../use-debounce';
import { useSearchTracks } from '../use-music-analysis';

export function useOptimizedSearch(initialQuery = '', debounceMs = 500) {
  const [query, setQuery] = useState(initialQuery);
  const debouncedQuery = useDebounce(query, debounceMs);
  const { data, isLoading, error, refetch } = useSearchTracks(
    debouncedQuery,
    20
  );

  const results = useMemo(() => data?.results || [], [data?.results]);
  const hasResults = results.length > 0;
  const hasQuery = query.length > 0;
  const isSearching = hasQuery && isLoading;

  return {
    query,
    setQuery,
    debouncedQuery,
    results,
    hasResults,
    hasQuery,
    isSearching,
    isLoading,
    error,
    refetch,
    total: data?.total || 0,
  };
}


