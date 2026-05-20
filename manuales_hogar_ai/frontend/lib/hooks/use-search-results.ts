import { useMemo } from 'react';
import { normalizeSearchResults } from '../utils/search-results';
import type { SearchResponse, SemanticSearchResponse } from '../types/api';
import { SEARCH_TABS } from '../constants';

interface SearchQueryState {
  data?: SearchResponse | SemanticSearchResponse;
  isLoading: boolean;
  error: Error | null;
}

interface UseSearchResultsOptions {
  simpleSearch: SearchQueryState;
  semanticSearch: SearchQueryState;
  advancedSearch: SearchQueryState;
  activeTab: string;
}

interface UseSearchResultsReturn {
  results: ReturnType<typeof normalizeSearchResults>;
  isLoading: boolean;
  error: Error | null;
}

export const useSearchResults = ({
  simpleSearch,
  semanticSearch,
  advancedSearch,
  activeTab,
}: UseSearchResultsOptions): UseSearchResultsReturn => {
  const { results, isLoading, error } = useMemo(() => {
    if (activeTab === SEARCH_TABS.SIMPLE) {
      return {
        results: simpleSearch.data,
        isLoading: simpleSearch.isLoading,
        error: simpleSearch.error,
      };
    }
    if (activeTab === SEARCH_TABS.SEMANTIC) {
      return {
        results: semanticSearch.data,
        isLoading: semanticSearch.isLoading,
        error: semanticSearch.error,
      };
    }
    return {
      results: advancedSearch.data,
      isLoading: advancedSearch.isLoading,
      error: advancedSearch.error,
    };
  }, [activeTab, simpleSearch.data, simpleSearch.isLoading, simpleSearch.error, semanticSearch.data, semanticSearch.isLoading, semanticSearch.error, advancedSearch.data, advancedSearch.isLoading, advancedSearch.error]);

  const normalizedResults = useMemo(
    () => normalizeSearchResults(results),
    [results]
  );

  return {
    results: normalizedResults,
    isLoading,
    error,
  };
};

