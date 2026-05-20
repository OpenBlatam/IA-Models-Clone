import {
  testApiConnection,
  validateApiConfig,
  getApiConnectionInfo,
} from '@/lib/api/connection-utils';
import { checkApiHealth } from '@/lib/api/client';
import { apiConfig } from '@/lib/config/app';
import { env } from '@/lib/config/env';

// Mock dependencies
jest.mock('@/lib/api/client', () => ({
  checkApiHealth: jest.fn(),
}));

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
    NODE_ENV: 'test',
  },
}));

const mockCheckApiHealth = checkApiHealth as jest.MockedFunction<
  typeof checkApiHealth
>;

describe('Connection Utils', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('testApiConnection', () => {
    it('should return success when API is healthy', async () => {
      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      const result = await testApiConnection();

      expect(result.success).toBe(true);
      expect(result.baseURL).toBe('http://localhost:8000/api/music');
      expect(result.message).toBe('API is reachable');
      expect(result.responseTime).toBeDefined();
      expect(result.responseTime).toBeGreaterThanOrEqual(0);
      expect(result.error).toBeUndefined();
    });

    it('should return failure when API is unhealthy', async () => {
      mockCheckApiHealth.mockResolvedValue({
        status: 'unhealthy',
        message: 'Connection failed',
        timestamp: Date.now(),
      });

      const result = await testApiConnection();

      expect(result.success).toBe(false);
      expect(result.message).toBe('Connection failed');
      expect(result.responseTime).toBeDefined();
    });

    it('should handle errors and return error message', async () => {
      const error = new Error('Network error');
      mockCheckApiHealth.mockRejectedValue(error);

      const result = await testApiConnection();

      expect(result.success).toBe(false);
      expect(result.message).toBe('Connection test failed');
      expect(result.error).toBe('Network error');
      expect(result.responseTime).toBeDefined();
    });

    it('should handle non-Error objects', async () => {
      mockCheckApiHealth.mockRejectedValue('String error');

      const result = await testApiConnection();

      expect(result.success).toBe(false);
      expect(result.error).toBe('Unknown error');
    });

    it('should measure response time', async () => {
      mockCheckApiHealth.mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  status: 'healthy',
                  message: 'API is reachable',
                  timestamp: Date.now(),
                }),
              100
            );
          })
      );

      const startTime = Date.now();
      const result = await testApiConnection();
      const endTime = Date.now();

      expect(result.responseTime).toBeDefined();
      expect(result.responseTime).toBeGreaterThanOrEqual(0);
      expect(result.responseTime).toBeLessThanOrEqual(endTime - startTime + 50); // Allow some margin
    });
  });

  describe('validateApiConfig', () => {
    it('should return valid when config is correct', () => {
      const result = validateApiConfig();

      expect(result.isValid).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    it('should detect missing base URL', () => {
      const originalBaseURL = apiConfig.music.baseURL;
      apiConfig.music.baseURL = '';

      const result = validateApiConfig();

      expect(result.isValid).toBe(false);
      expect(result.issues).toContain(
        'Music API base URL is not properly configured'
      );

      // Restore
      apiConfig.music.baseURL = originalBaseURL;
    });

    it('should detect invalid URL', () => {
      const originalBaseURL = apiConfig.music.baseURL;
      apiConfig.music.baseURL = 'not-a-valid-url';

      const result = validateApiConfig();

      expect(result.isValid).toBe(false);
      expect(result.issues).toContain('Music API base URL is not a valid URL');

      // Restore
      apiConfig.music.baseURL = originalBaseURL;
    });

    it('should detect invalid timeout', () => {
      const originalTimeout = apiConfig.music.timeout;
      apiConfig.music.timeout = 0;

      const result = validateApiConfig();

      expect(result.isValid).toBe(false);
      expect(result.issues).toContain('API timeout must be greater than 0');

      // Restore
      apiConfig.music.timeout = originalTimeout;
    });

    it('should detect negative retries', () => {
      const originalRetries = apiConfig.music.retries;
      apiConfig.music.retries = -1;

      const result = validateApiConfig();

      expect(result.isValid).toBe(false);
      expect(result.issues).toContain('API retries must be non-negative');

      // Restore
      apiConfig.music.retries = originalRetries;
    });

    it('should log config in development mode', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      const originalIsDev = env.IS_DEVELOPMENT;
      env.IS_DEVELOPMENT = true;

      validateApiConfig();

      expect(consoleSpy).toHaveBeenCalledWith(
        '[API Config]',
        expect.objectContaining({
          baseURL: expect.any(String),
          timeout: expect.any(Number),
          retries: expect.any(Number),
          environment: expect.any(String),
        })
      );

      // Restore
      env.IS_DEVELOPMENT = originalIsDev;
      consoleSpy.mockRestore();
    });

    it('should not log in production mode', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      const originalIsDev = env.IS_DEVELOPMENT;
      env.IS_DEVELOPMENT = false;

      validateApiConfig();

      expect(consoleSpy).not.toHaveBeenCalled();

      // Restore
      env.IS_DEVELOPMENT = originalIsDev;
      consoleSpy.mockRestore();
    });
  });

  describe('getApiConnectionInfo', () => {
    it('should return connection information', () => {
      const info = getApiConnectionInfo();

      expect(info).toEqual({
        baseURL: apiConfig.music.baseURL,
        timeout: apiConfig.music.timeout,
        retries: apiConfig.music.retries,
        environment: env.NODE_ENV,
      });
    });

    it('should return correct base URL', () => {
      const info = getApiConnectionInfo();
      expect(info.baseURL).toBe('http://localhost:8000/api/music');
    });

    it('should return correct timeout', () => {
      const info = getApiConnectionInfo();
      expect(info.timeout).toBe(10000);
    });

    it('should return correct retries', () => {
      const info = getApiConnectionInfo();
      expect(info.retries).toBe(3);
    });

    it('should return environment', () => {
      const info = getApiConnectionInfo();
      expect(info.environment).toBe('test');
    });
  });
});

