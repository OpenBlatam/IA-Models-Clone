import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { Category, AdvancedSearchRequest } from '../types/api';

export const useSemanticSearch = (
  query: string,
  category?: Category,
  limit = 10,
  threshold = 0.5
) => {
  return useQuery({
    queryKey: ['semantic-search', query, category, limit, threshold],
    queryFn: () => apiClient.semanticSearch(query, category, limit, threshold),
    enabled: query.length > 0,
  });
};

export const useAdvancedSearch = (request: AdvancedSearchRequest) => {
  return useQuery({
    queryKey: ['advanced-search', request],
    queryFn: () => apiClient.advancedSearch(request),
    enabled: !!request.query || !!request.category,
  });
};

