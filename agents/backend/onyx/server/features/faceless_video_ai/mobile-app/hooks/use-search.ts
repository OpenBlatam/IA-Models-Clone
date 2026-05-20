import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { searchService } from '@/services/search-service';
import type { SearchFilters } from '@/types/api';

export function useSearchVideos(
  query: string,
  filters?: SearchFilters,
  limit = 50,
  enabled = true
) {
  return useQuery({
    queryKey: ['search', 'videos', query, filters, limit],
    queryFn: () => searchService.searchVideos(query, filters, limit),
    enabled: enabled && query.length > 0,
    staleTime: 30000, // 30 seconds
  });
}

export function useSearchSuggestions(query: string, limit = 5, enabled = true) {
  return useQuery({
    queryKey: ['search', 'suggestions', query, limit],
    queryFn: () => searchService.getSearchSuggestions(query, limit),
    enabled: enabled && query.length > 0,
    staleTime: 60000, // 1 minute
  });
}

export function useClearSearchHistory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      // Clear search cache
      queryClient.removeQueries({ queryKey: ['search'] });
    },
  });
}


