import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type { TrendsResponse } from '../../types/api';

export class TrendsService {
  async getTrendsPopularity(
    timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term',
    limit = 20
  ): Promise<TrendsResponse> {
    const validTimeRanges = ['short_term', 'medium_term', 'long_term'];
    if (!validTimeRanges.includes(timeRange)) {
      throw new Error(`Time range must be one of: ${validTimeRanges.join(', ')}`);
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrendsResponse>(
      `${API_ENDPOINTS.TRENDS_POPULARITY}?time_range=${timeRange}&limit=${limit}`
    );
  }

  async getTrendsArtists(
    timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term',
    limit = 20
  ): Promise<TrendsResponse> {
    const validTimeRanges = ['short_term', 'medium_term', 'long_term'];
    if (!validTimeRanges.includes(timeRange)) {
      throw new Error(`Time range must be one of: ${validTimeRanges.join(', ')}`);
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrendsResponse>(
      `${API_ENDPOINTS.TRENDS_ARTISTS}?time_range=${timeRange}&limit=${limit}`
    );
  }
}

export const trendsService = new TrendsService();

