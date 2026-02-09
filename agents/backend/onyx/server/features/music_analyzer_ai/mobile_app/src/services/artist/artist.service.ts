import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type {
  ArtistComparisonRequest,
  ArtistComparison,
  ArtistEvolution,
} from '../../types/api';

export class ArtistService {
  async compareArtists(request: ArtistComparisonRequest): Promise<ArtistComparison> {
    if (!request.artist_ids || request.artist_ids.length === 0) {
      throw new Error('At least one artist ID is required');
    }
    if (request.artist_ids.length < 2) {
      throw new Error('At least two artist IDs are required for comparison');
    }
    return apiClient.post<ArtistComparison>(API_ENDPOINTS.ARTIST_COMPARE, request);
  }

  async getArtistEvolution(artistId: string): Promise<ArtistEvolution> {
    if (!artistId || artistId.trim().length === 0) {
      throw new Error('Artist ID is required');
    }
    return apiClient.get<ArtistEvolution>(
      `${API_ENDPOINTS.ARTIST_EVOLUTION}?artist_id=${artistId}`
    );
  }
}

export const artistService = new ArtistService();

