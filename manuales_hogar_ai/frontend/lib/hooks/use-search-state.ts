import { useState, useCallback } from 'react';
import type { Category } from '../types/api';

interface UseSearchStateReturn {
  query: string;
  category: Category | undefined;
  setQuery: (query: string) => void;
  setCategory: (category: Category | undefined) => void;
  reset: () => void;
}

export const useSearchState = (): UseSearchStateReturn => {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<Category | undefined>();

  const reset = useCallback((): void => {
    setQuery('');
    setCategory(undefined);
  }, []);

  return {
    query,
    category,
    setQuery,
    setCategory,
    reset,
  };
};

