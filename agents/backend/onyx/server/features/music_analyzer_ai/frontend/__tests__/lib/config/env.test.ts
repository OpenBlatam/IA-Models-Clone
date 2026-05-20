import { env } from '@/lib/config/env';

// Save original process.env
const originalEnv = process.env;

describe('Env Config', () => {
  beforeEach(() => {
    // Reset process.env
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('API URLs', () => {
    it('should use default values when env vars are not set', () => {
      delete process.env.NEXT_PUBLIC_MUSIC_API_URL;
      delete process.env.NEXT_PUBLIC_ROBOT_API_URL;

      // Re-import to get fresh config
      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.MUSIC_API_URL).toBe('http://localhost:8010');
      expect(freshEnv.ROBOT_API_URL).toBe('http://localhost:8010');
    });

    it('should use environment variables when set', () => {
      process.env.NEXT_PUBLIC_MUSIC_API_URL = 'http://custom-music-api:8080';
      process.env.NEXT_PUBLIC_ROBOT_API_URL = 'http://custom-robot-api:8080';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.MUSIC_API_URL).toBe('http://custom-music-api:8080');
      expect(freshEnv.ROBOT_API_URL).toBe('http://custom-robot-api:8080');
    });
  });

  describe('Environment Detection', () => {
    it('should detect development environment', () => {
      process.env.NODE_ENV = 'development';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.NODE_ENV).toBe('development');
      expect(freshEnv.IS_DEVELOPMENT).toBe(true);
      expect(freshEnv.IS_PRODUCTION).toBe(false);
    });

    it('should detect production environment', () => {
      process.env.NODE_ENV = 'production';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.NODE_ENV).toBe('production');
      expect(freshEnv.IS_DEVELOPMENT).toBe(false);
      expect(freshEnv.IS_PRODUCTION).toBe(true);
    });

    it('should default to development when NODE_ENV is not set', () => {
      delete process.env.NODE_ENV;

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.NODE_ENV).toBe('development');
    });
  });

  describe('Feature Flags', () => {
    it('should read voice commands feature flag', () => {
      process.env.NEXT_PUBLIC_ENABLE_VOICE_COMMANDS = 'true';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.ENABLE_VOICE_COMMANDS).toBe(true);
    });

    it('should default voice commands to false when not set', () => {
      delete process.env.NEXT_PUBLIC_ENABLE_VOICE_COMMANDS;

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.ENABLE_VOICE_COMMANDS).toBe(false);
    });

    it('should read offline mode feature flag', () => {
      process.env.NEXT_PUBLIC_ENABLE_OFFLINE_MODE = 'true';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.ENABLE_OFFLINE_MODE).toBe(true);
    });

    it('should read analytics feature flag', () => {
      process.env.NEXT_PUBLIC_ENABLE_ANALYTICS = 'true';

      jest.resetModules();
      const { env: freshEnv } = require('@/lib/config/env');

      expect(freshEnv.ENABLE_ANALYTICS).toBe(true);
    });
  });
});

