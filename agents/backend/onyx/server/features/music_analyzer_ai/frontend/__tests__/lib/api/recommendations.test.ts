import {
  getRecommendations,
  getContextualRecommendations,
  getRecommendationsByMood,
  getRecommendationsByActivity,
  getRecommendationsByTimeOfDay,
} from '@/lib/api/recommendations';
import { musicApiClient } from '@/lib/api/client';
import { ValidationError } from '@/lib/errors';

// Mock the API client
jest.mock('@/lib/api/client', () => ({
  musicApiClient: {
    get: jest.fn(),
    post: jest.fn(),
  },
}));

const mockMusicApiClient = musicApiClient as jest.Mocked<typeof musicApiClient>;

describe('Recommendations API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getRecommendations', () => {
    it('should get recommendations with default limit', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      const result = await getRecommendations('track-123');

      expect(mockMusicApiClient.get).toHaveBeenCalledWith(
        '/track/track-123/recommendations',
        {
          params: { limit: 20 },
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should get recommendations with custom limit', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      await getRecommendations('track-123', 10);

      expect(mockMusicApiClient.get).toHaveBeenCalledWith(
        '/track/track-123/recommendations',
        {
          params: { limit: 10 },
        }
      );
    });

    it('should throw ValidationError if trackId is empty', async () => {
      await expect(getRecommendations('')).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.get).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if limit is less than 1', async () => {
      await expect(getRecommendations('track-123', 0)).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.get).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if limit is greater than 50', async () => {
      await expect(getRecommendations('track-123', 51)).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.get).not.toHaveBeenCalled();
    });
  });

  describe('getContextualRecommendations', () => {
    it('should get contextual recommendations', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      const result = await getContextualRecommendations(
        'track-123',
        'workout',
        15
      );

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/contextual',
        {
          track_id: 'track-123',
          context: 'workout',
          limit: 15,
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should get contextual recommendations without context', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      await getContextualRecommendations('track-123');

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/contextual',
        {
          track_id: 'track-123',
          context: undefined,
          limit: 20,
        }
      );
    });

    it('should throw ValidationError if trackId is empty', async () => {
      await expect(
        getContextualRecommendations('', 'context')
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });
  });

  describe('getRecommendationsByMood', () => {
    it('should get recommendations by mood', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      const result = await getRecommendationsByMood('happy', 10);

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/mood',
        {
          mood: 'happy',
          limit: 10,
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should use default limit when not provided', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      await getRecommendationsByMood('happy');

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/mood',
        {
          mood: 'happy',
          limit: 20,
        }
      );
    });

    it('should throw ValidationError if mood is empty', async () => {
      await expect(getRecommendationsByMood('')).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });
  });

  describe('getRecommendationsByActivity', () => {
    it('should get recommendations by activity', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      const result = await getRecommendationsByActivity('running', 15);

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/activity',
        {
          activity: 'running',
          limit: 15,
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should throw ValidationError if activity is empty', async () => {
      await expect(getRecommendationsByActivity('')).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });
  });

  describe('getRecommendationsByTimeOfDay', () => {
    it('should get recommendations by time of day', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      const result = await getRecommendationsByTimeOfDay('morning', 10);

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/recommendations/time-of-day',
        {
          time_of_day: 'morning',
          limit: 10,
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should handle different times of day', async () => {
      const times = ['morning', 'afternoon', 'evening', 'night'];
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      for (const time of times) {
        await getRecommendationsByTimeOfDay(time);
        expect(mockMusicApiClient.post).toHaveBeenCalledWith(
          '/recommendations/time-of-day',
          {
            time_of_day: time,
            limit: 20,
          }
        );
      }
    });

    it('should throw ValidationError if timeOfDay is empty', async () => {
      await expect(getRecommendationsByTimeOfDay('')).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });
  });
});

