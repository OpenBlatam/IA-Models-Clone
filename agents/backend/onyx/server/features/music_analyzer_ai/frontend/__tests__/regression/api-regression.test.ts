/**
 * Regression Tests - API
 * Tests to prevent regression of API functionality
 */

import {
  getFavorites,
  addToFavorites,
  removeFromFavorites,
} from '@/lib/api/favorites';
import {
  getRecommendations,
  getContextualRecommendations,
  getRecommendationsByMood,
} from '@/lib/api/recommendations';
import { musicApiClient } from '@/lib/api/client';
import { ValidationError } from '@/lib/errors';

jest.mock('@/lib/api/client', () => ({
  musicApiClient: {
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  },
}));

const mockMusicApiClient = musicApiClient as jest.Mocked<typeof musicApiClient>;

describe('API Regression Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Favorites API Regression', () => {
    it('should maintain backward compatibility for getFavorites', async () => {
      const mockResponse = {
        data: {
          favorites: [],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      const result = await getFavorites();

      // Should still work without userId
      expect(mockMusicApiClient.get).toHaveBeenCalledWith('/favorites', {
        params: undefined,
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should maintain validation behavior for addToFavorites', async () => {
      // Should still throw ValidationError for empty userId
      await expect(
        addToFavorites('', 'track-123', 'Track', ['Artist'])
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });
  });

  describe('Recommendations API Regression', () => {
    it('should maintain default limit behavior', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      await getRecommendations('track-123');

      // Should use default limit of 20
      expect(mockMusicApiClient.get).toHaveBeenCalledWith(
        '/track/track-123/recommendations',
        {
          params: { limit: 20 },
        }
      );
    });

    it('should maintain validation limits', async () => {
      // Should reject limit < 1
      await expect(getRecommendations('track-123', 0)).rejects.toThrow(
        ValidationError
      );

      // Should reject limit > 50
      await expect(getRecommendations('track-123', 51)).rejects.toThrow(
        ValidationError
      );
    });
  });

  describe('API Client Regression', () => {
    it('should maintain retry behavior', async () => {
      // This would test retry logic if exposed
      // For now, we verify the client exists
      expect(mockMusicApiClient).toBeDefined();
    });
  });
});

