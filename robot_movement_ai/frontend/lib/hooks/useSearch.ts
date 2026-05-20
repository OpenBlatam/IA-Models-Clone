import { useState, useMemo, useCallback } from 'react';
import { fuzzySearch } from '@/lib/utils/search';

export interface UseSearchOptions<T> {
  initialQuery?: string;
  searchFields?: Array<keyof T | ((item: T) => string)>;
  fuzzy?: boolean;
  threshold?: number;
}

/**
 * Hook for search functionality
 */
export function useSearch<T>(
  data: T[],
  options: UseSearchOptions<T> = {}
): {
  query: string;
  setQuery: (query: string) => void;
  results: T[];
  clearSearch: () => void;
} {
  const {
    initialQuery = '',
    searchFields,
    fuzzy = false,
    threshold = 0.3,
  } = options;

  const [query, setQuery] = useState(initialQuery);

  const results = useMemo(() => {
    if (!query.trim()) return data;

    const lowerQuery = query.toLowerCase();

    if (fuzzy) {
      return data.filter((item) => {
        if (searchFields) {
          return searchFields.some((field) => {
            const value = typeof field === 'function' ? field(item) : item[field];
            const score = fuzzySearch(String(value), query);
            return score >= threshold;
          });
        }
        // Search in all string properties
        return Object.values(item as any).some((value) => {
          const score = fuzzySearch(String(value), query);
          return score >= threshold;
        });
      });
    }

    // Exact search
    return data.filter((item) => {
      if (searchFields) {
        return searchFields.some((field) => {
          const value = typeof field === 'function' ? field(item) : item[field];
          return String(value).toLowerCase().includes(lowerQuery);
        });
      }
      return Object.values(item as any).some((value) =>
        String(value).toLowerCase().includes(lowerQuery)
      );
    });
  }, [data, query, searchFields, fuzzy, threshold]);

  const clearSearch = useCallback(() => {
    setQuery('');
  }, []);

  return {
    query,
    setQuery,
    results,
    clearSearch,
  };
}



