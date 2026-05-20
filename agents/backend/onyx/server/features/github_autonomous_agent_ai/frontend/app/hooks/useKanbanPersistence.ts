import { useState, useEffect, useCallback } from 'react';

interface SavedFilter {
  name: string;
  filters: {
    searchQuery: string;
    selectedRepository: string;
    statusFilter: string;
    dateFilter: string;
    sortBy: string;
    sortOrder: string;
  };
}

export function useKanbanPersistence() {
  const [savedFilters, setSavedFilters] = useState<SavedFilter[]>(() => {
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('kanban-saved-filters');
        return saved ? JSON.parse(saved) : [];
      } catch {
        return [];
      }
    }
    return [];
  });

  // Persistir filtros guardados
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-saved-filters', JSON.stringify(savedFilters));
    }
  }, [savedFilters]);

  const saveCurrentFilters = useCallback((filters: SavedFilter['filters'], name: string) => {
    const newFilter: SavedFilter = {
      name,
      filters,
    };
    setSavedFilters((prev) => {
      const updated = [...prev, newFilter];
      if (typeof window !== 'undefined') {
        localStorage.setItem('kanban-saved-filters', JSON.stringify(updated));
      }
      return updated;
    });
    return newFilter;
  }, []);

  const applySavedFilter = useCallback((filter: SavedFilter) => {
    return filter.filters;
  }, []);

  const deleteSavedFilter = useCallback((index: number) => {
    setSavedFilters((prev) => {
      const updated = prev.filter((_, i) => i !== index);
      if (typeof window !== 'undefined') {
        localStorage.setItem('kanban-saved-filters', JSON.stringify(updated));
      }
      return updated;
    });
  }, []);

  return {
    savedFilters,
    saveCurrentFilters,
    applySavedFilter,
    deleteSavedFilter,
  };
}

