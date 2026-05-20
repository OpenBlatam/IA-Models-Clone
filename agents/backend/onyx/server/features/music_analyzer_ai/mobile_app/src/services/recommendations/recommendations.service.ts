import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type {
  RecommendationsResponse,
  ContextualRecommendationRequest,
} from '../../types/api';

export class RecommendationsService {
  async getRecommendations(trackId: string): Promise<RecommendationsResponse> {
    if (!trackId || trackId.trim().length === 0) {
      throw new Error('Track ID is required');
    }
    return apiClient.get<RecommendationsResponse>(
      API_ENDPOINTS.RECOMMENDATIONS(trackId)
    );
  }

  async getContextualRecommendations(
    request: ContextualRecommendationRequest
  ): Promise<RecommendationsResponse> {
    if (!request.context || request.context.trim().length === 0) {
      throw new Error('Context is required');
    }
    if (request.limit !== undefined && (request.limit < 1 || request.limit > 100)) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.post<RecommendationsResponse>(
      API_ENDPOINTS.CONTEXTUAL_RECOMMENDATIONS,
      request
    );
  }

  async getTimeOfDayRecommendations(
    timeOfDay: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    if (!timeOfDay || timeOfDay.trim().length === 0) {
      throw new Error('Time of day is required');
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<RecommendationsResponse>(
      `${API_ENDPOINTS.TIME_OF_DAY_RECOMMENDATIONS}?time_of_day=${timeOfDay}&limit=${limit}`
    );
  }

  async getActivityRecommendations(
    activity: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    if (!activity || activity.trim().length === 0) {
      throw new Error('Activity is required');
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<RecommendationsResponse>(
      `${API_ENDPOINTS.ACTIVITY_RECOMMENDATIONS}?activity=${activity}&limit=${limit}`
    );
  }

  async getMoodRecommendations(
    mood: string,
    limit = 20
  ): Promise<RecommendationsResponse> {
    if (!mood || mood.trim().length === 0) {
      throw new Error('Mood is required');
    }
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<RecommendationsResponse>(
      `${API_ENDPOINTS.MOOD_RECOMMENDATIONS}?mood=${mood}&limit=${limit}`
    );
  }
}

export const recommendationsService = new RecommendationsService();

