'use client';

import { useState, useMemo } from 'react';
import { useDebounce } from './useDebounce';
import type { TaskListItem } from '@/types/api';

interface SearchOptions {
  fields?: string[];
  debounceMs?: number;
}

export function useSearch<T extends TaskListItem>(
  items: T[],
  options: SearchOptions = {}
) {
  const { fields = ['query_preview', 'task_id'], debounceMs = 300 } = options;
  const [searchQuery, setSearchQuery] = useState('');
  const debouncedQuery = useDebounce(searchQuery, debounceMs);

  const filteredItems = useMemo(() => {
    if (!debouncedQuery.trim()) return items;

    const query = debouncedQuery.toLowerCase();
    return items.filter((item) => {
      return fields.some((field) => {
        const value = (item as any)[field];
        return value?.toLowerCase().includes(query);
      });
    });
  }, [items, debouncedQuery, fields]);

  return {
    searchQuery,
    setSearchQuery,
    filteredItems,
    debouncedQuery,
  };
}

