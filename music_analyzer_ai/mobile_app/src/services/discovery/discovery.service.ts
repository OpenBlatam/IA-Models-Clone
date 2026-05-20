import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type { TrackSearchResponse } from '../../types/api';

export class DiscoveryService {
  async discoverSimilarArtists(
    artistId: string,
    limit = 20
  ): Promise<TrackSearchResponse> {
    if (!artistId || artistId.trim().length === 0) {
      throw new Error('Artist ID is required');
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrackSearchResponse>(
      `${API_ENDPOINTS.DISCOVERY_SIMILAR_ARTISTS}?artist_id=${artistId}&limit=${limit}`
    );
  }

  async discoverUnderground(limit = 20): Promise<TrackSearchResponse> {
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrackSearchResponse>(
      `${API_ENDPOINTS.DISCOVERY_UNDERGROUND}?limit=${limit}`
    );
  }

  async discoverMoodTransition(
    fromMood: string,
    toMood: string,
    limit = 20
  ): Promise<TrackSearchResponse> {
    if (!fromMood || fromMood.trim().length === 0) {
      throw new Error('From mood is required');
    }
    if (!toMood || toMood.trim().length === 0) {
      throw new Error('To mood is required');
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrackSearchResponse>(
      `${API_ENDPOINTS.DISCOVERY_MOOD_TRANSITION}?from_mood=${fromMood}&to_mood=${toMood}&limit=${limit}`
    );
  }

  async discoverFresh(limit = 20): Promise<TrackSearchResponse> {
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<TrackSearchResponse>(
      `${API_ENDPOINTS.DISCOVERY_FRESH}?limit=${limit}`
    );
  }
}

export const discoveryService = new DiscoveryService();

