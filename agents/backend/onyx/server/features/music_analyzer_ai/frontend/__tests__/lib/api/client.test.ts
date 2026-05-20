import axios, { AxiosError } from 'axios';
import {
  createMusicApiClient,
  checkApiHealth,
  requestWithRetry,
} from '@/lib/api/client';
import { ApiError, NetworkError } from '@/lib/errors';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock config
jest.mock('@/lib/config/app', () => ({
  apiConfig: {
    music: {
      baseURL: 'http://localhost:8000/api/music',
      timeout: 10000,
      retries: 3,
    },
  },
}));

jest.mock('@/lib/config/env', () => ({
  env: {
    IS_DEVELOPMENT: false,
  },
}));

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('createMusicApiClient', () => {
    it('should create axios instance with correct config', () => {
      const client = createMusicApiClient();
      expect(client).toBeDefined();
      expect(client.defaults.baseURL).toBe('http://localhost:8000/api/music');
      expect(client.defaults.timeout).toBe(10000);
    });

    it('should add request interceptor', async () => {
      const client = createMusicApiClient();
      const mockGet = jest.fn().mockResolvedValue({ data: {} });
      client.get = mockGet;

      await client.get('/test');

      expect(mockGet).toHaveBeenCalled();
    });
  });

  describe('checkApiHealth', () => {
    it('should return healthy status when API is reachable', async () => {
      const mockClient = {
        get: jest.fn().mockResolvedValue({ status: 200 }),
      };
      jest.spyOn(require('@/lib/api/client'), 'musicApiClient').mockImplementation(() => mockClient);

      const result = await checkApiHealth();

      expect(result.status).toBe('healthy');
      expect(result.message).toBe('API is reachable');
      expect(result.timestamp).toBeDefined();
    });

    it('should return unhealthy status when API is unreachable', async () => {
      const mockClient = {
        get: jest.fn().mockRejectedValue(new Error('Network error')),
      };
      jest.spyOn(require('@/lib/api/client'), 'musicApiClient').mockImplementation(() => mockClient);

      const result = await checkApiHealth();

      expect(result.status).toBe('unhealthy');
      expect(result.message).toBeDefined();
    });
  });

  describe('requestWithRetry', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should retry failed requests', async () => {
      let attemptCount = 0;
      const requestFn = jest.fn().mockImplementation(() => {
        attemptCount++;
        if (attemptCount < 3) {
          throw new Error('Network error');
        }
        return Promise.resolve({ data: 'success' });
      });

      const promise = requestWithRetry(requestFn, { retries: 2 });

      // Fast-forward timers for retries
      jest.advanceTimersByTime(10000);

      const result = await promise;

      expect(requestFn).toHaveBeenCalledTimes(3);
      expect(result).toEqual({ data: 'success' });
    });

    it('should throw error after max retries', async () => {
      const requestFn = jest.fn().mockRejectedValue(new Error('Persistent error'));

      const promise = requestWithRetry(requestFn, { retries: 2 });

      jest.advanceTimersByTime(10000);

      await expect(promise).rejects.toThrow('Persistent error');
      expect(requestFn).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });

    it('should not retry on non-retryable errors', async () => {
      const apiError = new ApiError('Not found', 404);
      const requestFn = jest.fn().mockRejectedValue(apiError);

      const promise = requestWithRetry(requestFn, {
        retries: 2,
        retryCondition: (error) => {
          if (error instanceof ApiError) {
            return error.statusCode !== undefined && error.statusCode >= 500;
          }
          return true;
        },
      });

      jest.advanceTimersByTime(1000);

      await expect(promise).rejects.toThrow('Not found');
      expect(requestFn).toHaveBeenCalledTimes(1); // No retries for 404
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const mockClient = createMusicApiClient();
      const networkError = new AxiosError('Network Error');
      networkError.code = 'ECONNABORTED';

      mockClient.get = jest.fn().mockRejectedValue(networkError);

      await expect(mockClient.get('/test')).rejects.toThrow(NetworkError);
    });

    it('should handle 404 errors', async () => {
      const mockClient = createMusicApiClient();
      const notFoundError = new AxiosError('Not Found');
      notFoundError.response = {
        status: 404,
        data: { detail: 'Resource not found' },
      } as any;

      mockClient.get = jest.fn().mockRejectedValue(notFoundError);

      await expect(mockClient.get('/test')).rejects.toThrow(ApiError);
    });

    it('should handle 500 errors', async () => {
      const mockClient = createMusicApiClient();
      const serverError = new AxiosError('Server Error');
      serverError.response = {
        status: 500,
        data: { detail: 'Internal server error' },
      } as any;

      mockClient.get = jest.fn().mockRejectedValue(serverError);

      await expect(mockClient.get('/test')).rejects.toThrow(ApiError);
    });
  });
});

