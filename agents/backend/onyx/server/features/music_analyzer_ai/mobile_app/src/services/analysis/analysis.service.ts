import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type { TrackAnalysis, TrackComparison, ExportResponse } from '../../types/api';

export interface AnalyzeTrackRequest {
  track_id?: string;
  track_name?: string;
  include_coaching?: boolean;
}

export interface CompareTracksRequest {
  track_ids: string[];
}

export class AnalysisService {
  async analyzeTrack(request: AnalyzeTrackRequest): Promise<TrackAnalysis> {
    if (!request.track_id && !request.track_name) {
      throw new Error('Either track_id or track_name is required');
    }
    return apiClient.post<TrackAnalysis>(API_ENDPOINTS.ANALYZE, request);
  }

  async analyzeTrackById(
    trackId: string,
    includeCoaching = true
  ): Promise<TrackAnalysis> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    return apiClient.get<TrackAnalysis>(
      `${API_ENDPOINTS.ANALYZE_BY_ID(trackId)}?include_coaching=${includeCoaching}`
    );
  }

  async compareTracks(request: CompareTracksRequest): Promise<TrackComparison> {
    if (!request.track_ids || request.track_ids.length === 0) {
      throw new Error('At least one track ID is required');
    }
    if (request.track_ids.length < 2) {
      throw new Error('At least two track IDs are required for comparison');
    }
    return apiClient.post<TrackComparison>(API_ENDPOINTS.COMPARE, request);
  }

  async getCoaching(request: AnalyzeTrackRequest): Promise<TrackAnalysis> {
    if (!request.track_id && !request.track_name) {
      throw new Error('Either track_id or track_name is required');
    }
    return apiClient.post<TrackAnalysis>(API_ENDPOINTS.COACHING, request);
  }

  async exportAnalysis(
    trackId: string,
    format: 'json' | 'text' | 'markdown' = 'json',
    includeCoaching = true
  ): Promise<ExportResponse> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    const validFormats = ['json', 'text', 'markdown'];
    if (!validFormats.includes(format)) {
      throw new Error(`Format must be one of: ${validFormats.join(', ')}`);
    }
    const separator = API_ENDPOINTS.EXPORT(trackId, format).includes('?') ? '&' : '?';
    return apiClient.post<ExportResponse>(
      `${API_ENDPOINTS.EXPORT(trackId, format)}${separator}include_coaching=${includeCoaching}`
    );
  }
}

export const analysisService = new AnalysisService();

