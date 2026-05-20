import { appConfig, apiConfig, uiConfig, performanceConfig } from '@/lib/config/app';

// Mock env
jest.mock('@/lib/config/env', () => ({
  env: {
    MUSIC_API_URL: 'http://localhost:8010',
    ROBOT_API_URL: 'http://localhost:8010',
    IS_PRODUCTION: false,
  },
}));

describe('App Config', () => {
  describe('appConfig', () => {
    it('should have correct application metadata', () => {
      expect(appConfig.name).toBe('Blatam Academy');
      expect(appConfig.description).toBe('Music Analyzer AI & Robot Movement AI Platform');
      expect(appConfig.version).toBe('1.0.0');
      expect(appConfig.author).toBe('Blatam Academy');
    });

    it('should have keywords array', () => {
      expect(appConfig.keywords).toContain('music analysis');
      expect(appConfig.keywords).toContain('AI');
      expect(appConfig.keywords).toContain('music analyzer');
    });

    it('should have correct URL for development', () => {
      expect(appConfig.url).toBe('http://localhost:3000');
    });
  });

  describe('apiConfig', () => {
    it('should have correct music API config', () => {
      expect(apiConfig.music.baseURL).toBe('http://localhost:8010/music');
      expect(apiConfig.music.timeout).toBe(30000);
      expect(apiConfig.music.retries).toBe(2);
    });

    it('should have correct robot API config', () => {
      expect(apiConfig.robot.baseURL).toBe('http://localhost:8010/robot');
      expect(apiConfig.robot.timeout).toBe(30000);
      expect(apiConfig.robot.retries).toBe(2);
    });
  });

  describe('uiConfig', () => {
    it('should have correct theme config', () => {
      expect(uiConfig.theme.default).toBe('dark');
      expect(uiConfig.theme.storageKey).toBe('app-theme');
    });

    it('should have correct toast config', () => {
      expect(uiConfig.toast.duration).toBe(4000);
      expect(uiConfig.toast.position).toBe('top-right');
    });

    it('should have correct pagination config', () => {
      expect(uiConfig.pagination.defaultPageSize).toBe(20);
      expect(uiConfig.pagination.maxPageSize).toBe(100);
    });
  });

  describe('performanceConfig', () => {
    it('should have correct debounce config', () => {
      expect(performanceConfig.debounce.search).toBe(500);
      expect(performanceConfig.debounce.input).toBe(300);
      expect(performanceConfig.debounce.scroll).toBe(100);
    });

    it('should have correct cache config', () => {
      expect(performanceConfig.cache.staleTime).toBe(60000);
      expect(performanceConfig.cache.gcTime).toBe(300000);
    });
  });
});

