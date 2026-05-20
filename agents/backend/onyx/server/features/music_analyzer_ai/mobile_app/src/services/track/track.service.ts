import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type {
  Track,
  TrackSearchRequest,
  TrackSearchResponse,
  AudioFeatures,
  AudioAnalysis,
} from '../../types/api';

export class TrackService {
  async searchTracks(request: TrackSearchRequest): Promise<TrackSearchResponse> {
    if (!request.query || request.query.trim().length === 0) {
      throw new Error('Search query is required');
    }
    if (request.limit !== undefined && (request.limit < 1 || request.limit > 100)) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.post<TrackSearchResponse>(API_ENDPOINTS.SEARCH, request);
  }

  async getTrackInfo(trackId: string): Promise<Track> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    return apiClient.get<Track>(API_ENDPOINTS.TRACK_INFO(trackId));
  }

  async getAudioFeatures(trackId: string): Promise<AudioFeatures> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    return apiClient.get<AudioFeatures>(API_ENDPOINTS.AUDIO_FEATURES(trackId));
  }

  async getAudioAnalysis(trackId: string): Promise<AudioAnalysis> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    return apiClient.get<AudioAnalysis>(API_ENDPOINTS.AUDIO_ANALYSIS(trackId));
  }
}

export const trackService = new TrackService();

