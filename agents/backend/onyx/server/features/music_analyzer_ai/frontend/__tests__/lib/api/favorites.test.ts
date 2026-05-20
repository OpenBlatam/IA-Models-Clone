import {
  getFavorites,
  addToFavorites,
  removeFromFavorites,
} from '@/lib/api/favorites';
import { musicApiClient } from '@/lib/api/client';
import { ValidationError } from '@/lib/errors';

// Mock the API client
jest.mock('@/lib/api/client', () => ({
  musicApiClient: {
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  },
}));

const mockMusicApiClient = musicApiClient as jest.Mocked<typeof musicApiClient>;

describe('Favorites API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getFavorites', () => {
    it('should get favorites without userId', async () => {
      const mockResponse = {
        data: {
          favorites: [],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      const result = await getFavorites();

      expect(mockMusicApiClient.get).toHaveBeenCalledWith('/favorites', {
        params: undefined,
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should get favorites with userId', async () => {
      const mockResponse = {
        data: {
          favorites: [
            {
              id: '1',
              name: 'Favorite Track',
            },
          ],
        },
      };

      mockMusicApiClient.get.mockResolvedValue(mockResponse);

      const result = await getFavorites('user-123');

      expect(mockMusicApiClient.get).toHaveBeenCalledWith('/favorites', {
        params: { user_id: 'user-123' },
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('addToFavorites', () => {
    it('should add track to favorites', async () => {
      const mockResponse = {
        data: {
          success: true,
          message: 'Track added to favorites',
        },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      const result = await addToFavorites(
        'user-123',
        'track-456',
        'Track Name',
        ['Artist 1', 'Artist 2']
      );

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/favorites',
        null,
        {
          params: {
            user_id: 'user-123',
            track_id: 'track-456',
            track_name: 'Track Name',
            artists: 'Artist 1,Artist 2',
          },
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should throw ValidationError if userId is empty', async () => {
      await expect(
        addToFavorites('', 'track-456', 'Track Name', ['Artist'])
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if trackId is empty', async () => {
      await expect(
        addToFavorites('user-123', '', 'Track Name', ['Artist'])
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if trackName is empty', async () => {
      await expect(
        addToFavorites('user-123', 'track-456', '', ['Artist'])
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if artists array is empty', async () => {
      await expect(
        addToFavorites('user-123', 'track-456', 'Track Name', [])
      ).rejects.toThrow(ValidationError);

      expect(mockMusicApiClient.post).not.toHaveBeenCalled();
    });

    it('should handle single artist', async () => {
      const mockResponse = {
        data: { success: true },
      };

      mockMusicApiClient.post.mockResolvedValue(mockResponse);

      await addToFavorites('user-123', 'track-456', 'Track Name', ['Artist']);

      expect(mockMusicApiClient.post).toHaveBeenCalledWith(
        '/favorites',
        null,
        expect.objectContaining({
          params: expect.objectContaining({
            artists: 'Artist',
          }),
        })
      );
    });
  });

  describe('removeFromFavorites', () => {
    it('should remove track from favorites', async () => {
      const mockResponse = {
        data: {
          success: true,
          message: 'Track removed from favorites',
        },
      };

      mockMusicApiClient.delete.mockResolvedValue(mockResponse);

      const result = await removeFromFavorites('user-123', 'track-456');

      expect(mockMusicApiClient.delete).toHaveBeenCalledWith(
        '/favorites/track-456',
        {
          params: { user_id: 'user-123' },
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should throw ValidationError if userId is empty', async () => {
      await expect(removeFromFavorites('', 'track-456')).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.delete).not.toHaveBeenCalled();
    });

    it('should throw ValidationError if trackId is empty', async () => {
      await expect(removeFromFavorites('user-123', '')).rejects.toThrow(
        ValidationError
      );

      expect(mockMusicApiClient.delete).not.toHaveBeenCalled();
    });
  });
});

