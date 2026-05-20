import type { ManualListItem, SearchResponse, SemanticSearchResponse } from '../types/api';

export const normalizeSearchResults = (
  results: SearchResponse | SemanticSearchResponse | undefined
): ManualListItem[] | undefined => {
  if (!results || !('results' in results)) {
    return undefined;
  }

  if ('similarity' in results.results[0]) {
    return (results as SemanticSearchResponse).results.map((r) => r.manual);
  }

  return (results as SearchResponse).results;
};

