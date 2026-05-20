import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type { SearchResult, SearchFilters } from '@/types/api';

export const searchService = {
  async searchVideos(
    query: string,
    filters?: SearchFilters,
    limit = 50
  ): Promise<{ results: SearchResult[]; total: number }> {
    const params = new URLSearchParams({
      q: query,
      limit: limit.toString(),
    });

    if (filters?.status) {
      params.append('status', filters.status);
    }
    if (filters?.tags) {
      params.append('tags', filters.tags.join(','));
    }
    if (filters?.date_from) {
      params.append('date_from', filters.date_from);
    }
    if (filters?.date_to) {
      params.append('date_to', filters.date_to);
    }

    return apiClient.get<{ results: SearchResult[]; total: number }>(
      `${API_ENDPOINTS.SEARCH.VIDEOS}?${params.toString()}`
    );
  },

  async getSearchSuggestions(query: string, limit = 5): Promise<{ suggestions: string[] }> {
    return apiClient.get<{ suggestions: string[] }>(
      `${API_ENDPOINTS.SEARCH.SUGGESTIONS}?q=${encodeURIComponent(query)}&limit=${limit}`
    );
  },
};


