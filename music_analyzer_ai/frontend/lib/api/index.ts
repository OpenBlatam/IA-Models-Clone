/**
 * Main API service module that exports all API functions.
 * This module provides a clean interface for all music analyzer API endpoints.
 */

// Re-export types
export * from './types';

// Re-export error types
export { ApiError, NetworkError, ValidationError } from '../errors';

// Track-related endpoints
export {
  searchTracks,
  analyzeTrack,
  getTrackInfo,
  getAudioFeatures,
  getAudioAnalysis,
} from './tracks';

// Comparison endpoints
export { compareTracks, compareTracksML } from './comparison';

// Recommendations endpoints
export {
  getRecommendations,
  getContextualRecommendations,
  getRecommendationsByMood,
  getRecommendationsByActivity,
  getRecommendationsByTimeOfDay,
} from './recommendations';

// Favorites endpoints
export {
  getFavorites,
  addToFavorites,
  removeFromFavorites,
} from './favorites';

// Re-export client for advanced usage
export {
  musicApiClient,
  checkApiHealth,
  requestWithRetry,
} from './client';

// Re-export connection utilities
export {
  testApiConnection,
  validateApiConfig,
  getApiConnectionInfo,
} from './connection-utils';
export type { ConnectionTestResult } from './connection-utils';

/**
 * Legacy API service object for backward compatibility.
 * @deprecated Use individual exported functions instead
 */
export const musicApiService = {
  searchTracks,
  analyzeTrack,
  getTrackInfo,
  getAudioFeatures,
  getAudioAnalysis,
  compareTracks,
  compareTracksML,
  getRecommendations,
  getContextualRecommendations,
  getRecommendationsByMood,
  getRecommendationsByActivity,
  getRecommendationsByTimeOfDay,
  getFavorites,
  addToFavorites,
  removeFromFavorites,
  // Additional endpoints can be added here as needed
  // For now, keeping the most commonly used ones
  healthCheck: async (): Promise<unknown> => {
    const { musicApiClient } = await import('./client');
    const response = await musicApiClient.get('/health');
    return response.data;
  },
  getHistory: async (userId?: string, limit: number = 50): Promise<unknown> => {
    const { musicApiClient } = await import('./client');
    const response = await musicApiClient.get('/history', {
      params: { user_id: userId, limit },
    });
    return response.data;
  },
  getAnalytics: async (): Promise<unknown> => {
    const { musicApiClient } = await import('./client');
    const response = await musicApiClient.get('/analytics');
    return response.data;
  },
  exportAnalysis: async (
    trackId: string,
    format: 'json' | 'text' | 'markdown' = 'json',
    includeCoaching: boolean = true
  ): Promise<unknown> => {
    const { musicApiClient } = await import('./client');
    const response = await musicApiClient.post(
      `/export/${trackId}`,
      null,
      {
        params: {
          format,
          include_coaching: includeCoaching,
        },
      }
    );
    return response.data;
  },
};

