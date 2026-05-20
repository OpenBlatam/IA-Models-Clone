'use client';

import { useState, useMemo } from 'react';
import type { FilterParams } from '@/types/common';

interface UseFiltersOptions<T> {
  items: T[];
  filters: FilterParams;
}

export function useFilters<T extends Record<string, any>>({
  items,
  filters,
}: UseFiltersOptions<T>) {
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      // Status filter
      if (filters.status && filters.status.length > 0) {
        if (!filters.status.includes(item.status)) return false;
      }

      // Date range filter
      if (filters.dateRange) {
        const itemDate = new Date(item.created_at || item.createdAt || Date.now());
        if (filters.dateRange.start && itemDate < filters.dateRange.start) return false;
        if (filters.dateRange.end && itemDate > filters.dateRange.end) return false;
      }

      // Business area filter
      if (filters.businessArea && filters.businessArea.length > 0) {
        const itemArea = item.business_area || item.businessArea;
        if (!filters.businessArea.includes(itemArea)) return false;
      }

      // Tags filter
      if (filters.tags && filters.tags.length > 0) {
        const itemTags = item.tags || [];
        const hasMatchingTag = filters.tags.some((tag) => itemTags.includes(tag));
        if (!hasMatchingTag) return false;
      }

      return true;
    });
  }, [items, filters]);

  return { filteredItems };
}

