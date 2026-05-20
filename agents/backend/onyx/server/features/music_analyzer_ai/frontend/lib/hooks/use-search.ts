/**
 * Custom hook for search functionality.
 * Provides convenient search state and filtering.
 */

import { useState, useMemo, useCallback } from 'react';
import {
  searchText,
  fuzzySearch,
  filterBySearch,
  sortByRelevance,
} from '../utils/search';

/**
 * Options for useSearch hook.
 */
export interface UseSearchOptions<T> {
  items: T[];
  getSearchText: (item: T) => string;
  fuzzy?: boolean;
  sortByRelevance?: boolean;
  debounceMs?: number;
}

/**
 * Return type for useSearch hook.
 */
export interface UseSearchReturn<T> {
  query: string;
  setQuery: (query: string) => void;
  results: T[];
  isSearching: boolean;
  clearSearch: () => void;
}

/**
 * Custom hook for search functionality.
 * Provides convenient search with filtering and sorting.
 *
 * @param options - Hook options
 * @returns Search state and results
 */
export function useSearch<T>(
  options: UseSearchOptions<T>
): UseSearchReturn<T> {
  const {
    items,
    getSearchText,
    fuzzy = false,
    sortByRelevance: shouldSort = false,
  } = options;

  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Debounce query
  useMemo(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, options.debounceMs || 300);

    return () => clearTimeout(timer);
  }, [query, options.debounceMs]);

  const results = useMemo(() => {
    if (!debouncedQuery.trim()) {
      return items;
    }

    const searchFn = fuzzy ? fuzzySearch : searchText;
    const filtered = items.filter((item) => {
      const text = getSearchText(item);
      return searchFn(text, debouncedQuery);
    });

    if (shouldSort) {
      return sortByRelevance(filtered, debouncedQuery, getSearchText);
    }

    return filtered;
  }, [items, debouncedQuery, getSearchText, fuzzy, shouldSort]);

  const clearSearch = useCallback(() => {
    setQuery('');
    setDebouncedQuery('');
  }, []);

  return {
    query,
    setQuery,
    results,
    isSearching: debouncedQuery.trim().length > 0,
    clearSearch,
  };
}

