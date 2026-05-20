import axios from 'axios';
import { musicApiService } from '@/lib/api/music-api';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock axios.create
const mockAxiosInstance = {
  post: jest.fn(),
  get: jest.fn(),
};

mockedAxios.create = jest.fn(() => mockAxiosInstance as any);

describe('Music API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('searchTracks', () => {
    it('should search tracks with query and limit', async () => {
      const mockResponse = {
        data: {
          success: true,
          query: 'test',
          results: [],
          total: 0,
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await musicApiService.searchTracks('test', 10);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/search', {
        query: 'test',
        limit: 10,
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should use default limit when not provided', async () => {
      const mockResponse = {
        data: {
          success: true,
          query: 'test',
          results: [],
          total: 0,
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      await musicApiService.searchTracks('test');

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/search', {
        query: 'test',
        limit: 10,
      });
    });
  });

  describe('analyzeTrack', () => {
    it('should analyze track with trackId', async () => {
      const mockResponse = {
        data: {
          success: true,
          track_basic_info: {
            name: 'Test Track',
            artists: ['Artist'],
            album: 'Album',
            duration_seconds: 200,
          },
          musical_analysis: {},
          technical_analysis: {},
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await musicApiService.analyzeTrack('track-id');

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze', {
        track_id: 'track-id',
        include_coaching: true,
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should analyze track with trackName', async () => {
      const mockResponse = {
        data: {
          success: true,
          track_basic_info: {},
          musical_analysis: {},
          technical_analysis: {},
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      await musicApiService.analyzeTrack(undefined, 'Track Name');

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze', {
        track_name: 'Track Name',
        include_coaching: true,
      });
    });

    it('should include coaching when specified', async () => {
      const mockResponse = {
        data: {
          success: true,
          track_basic_info: {},
          musical_analysis: {},
          technical_analysis: {},
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      await musicApiService.analyzeTrack('track-id', undefined, false);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze', {
        track_id: 'track-id',
        include_coaching: false,
      });
    });
  });

  describe('getTrackInfo', () => {
    it('should get track info', async () => {
      const mockResponse = {
        data: {
          id: 'track-id',
          name: 'Test Track',
        },
      };

      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      const result = await musicApiService.getTrackInfo('track-id');

      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/track/track-id/info'
      );
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('getAudioFeatures', () => {
    it('should get audio features', async () => {
      const mockResponse = {
        data: {
          energy: 0.8,
          danceability: 0.7,
        },
      };

      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      const result = await musicApiService.getAudioFeatures('track-id');

      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/track/track-id/audio-features'
      );
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('getAudioAnalysis', () => {
    it('should get audio analysis', async () => {
      const mockResponse = {
        data: {
          bars: [],
          beats: [],
        },
      };

      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      const result = await musicApiService.getAudioAnalysis('track-id');

      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/track/track-id/audio-analysis'
      );
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('compareTracks', () => {
    it('should compare tracks', async () => {
      const mockResponse = {
        data: {
          success: true,
          comparison: {
            key_signatures: {
              all_same: true,
              keys: ['C'],
              most_common: 'C',
            },
            tempos: {
              average: 120,
              min: 100,
              max: 140,
              range: 40,
            },
          },
          similarities: [],
          differences: [],
          recommendations: [],
        },
      };

      mockAxiosInstance.post.mockResolvedValue(mockResponse);

      const result = await musicApiService.compareTracks([
        'track-1',
        'track-2',
      ]);

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/compare', {
        track_ids: ['track-1', 'track-2'],
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('getRecommendations', () => {
    it('should get recommendations with default limit', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      await musicApiService.getRecommendations('track-id');

      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/track/track-id/recommendations',
        { params: { limit: 20 } }
      );
    });

    it('should get recommendations with custom limit', async () => {
      const mockResponse = {
        data: {
          tracks: [],
        },
      };

      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      await musicApiService.getRecommendations('track-id', 10);

      expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        '/track/track-id/recommendations',
        { params: { limit: 10 } }
      );
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const error = new Error('Network error');
      mockAxiosInstance.post.mockRejectedValue(error);

      await expect(
        musicApiService.searchTracks('test')
      ).rejects.toThrow('Network error');
    });

    it('should handle API errors', async () => {
      const error = {
        response: {
          status: 404,
          data: { message: 'Not found' },
        },
      };
      mockAxiosInstance.post.mockRejectedValue(error);

      await expect(
        musicApiService.searchTracks('test')
      ).rejects.toEqual(error);
    });
  });
});
