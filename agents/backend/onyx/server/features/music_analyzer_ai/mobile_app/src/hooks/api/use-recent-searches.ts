import { useCallback } from 'react';
import { useMusicStore } from '../../stores/music';
import type { Track } from '../../types/api';

export function useRecentSearches() {
  const recentSearches = useMusicStore((state) => state.recentSearches);
  const addRecentSearch = useMusicStore((state) => state.addRecentSearch);
  const clearRecentSearches = useMusicStore(
    (state) => state.clearRecentSearches
  );

  const hasRecentSearches = recentSearches.length > 0;

  return {
    recentSearches,
    addRecentSearch,
    clearRecentSearches,
    hasRecentSearches,
    recentSearchesCount: recentSearches.length,
  };
}


