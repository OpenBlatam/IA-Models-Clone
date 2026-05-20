import {
  API_ENDPOINTS,
  QUERY_KEYS,
  ROUTES,
  STORAGE_KEYS,
  PAGINATION,
  DEBOUNCE_DELAYS,
  TIMEOUTS,
  VALIDATION_LIMITS,
  FEATURES,
} from '@/lib/constants';

describe('Constants', () => {
  describe('API_ENDPOINTS', () => {
    it('should have correct music endpoints', () => {
      expect(API_ENDPOINTS.MUSIC.BASE).toBe('/music');
      expect(API_ENDPOINTS.MUSIC.SEARCH).toBe('/music/search');
      expect(API_ENDPOINTS.MUSIC.ANALYZE).toBe('/music/analyze');
      expect(API_ENDPOINTS.MUSIC.COMPARE).toBe('/music/compare');
      expect(API_ENDPOINTS.MUSIC.FAVORITES).toBe('/music/favorites');
      expect(API_ENDPOINTS.MUSIC.HEALTH).toBe('/music/health');
    });

    it('should generate track endpoint with ID', () => {
      expect(API_ENDPOINTS.MUSIC.TRACK('track-123')).toBe('/music/track/track-123');
    });

    it('should generate recommendations endpoint with ID', () => {
      expect(API_ENDPOINTS.MUSIC.RECOMMENDATIONS('track-123')).toBe(
        '/music/track/track-123/recommendations'
      );
    });

    it('should have correct robot endpoints', () => {
      expect(API_ENDPOINTS.ROBOT.BASE).toBe('/robot');
      expect(API_ENDPOINTS.ROBOT.HEALTH).toBe('/robot/health');
    });
  });

  describe('QUERY_KEYS', () => {
    it('should generate music search query key', () => {
      const key = QUERY_KEYS.MUSIC.SEARCH('test query');
      expect(key).toEqual(['music', 'search', 'test query']);
    });

    it('should generate track query key', () => {
      const key = QUERY_KEYS.MUSIC.TRACK('track-123');
      expect(key).toEqual(['music', 'track', 'track-123']);
    });

    it('should generate analysis query key', () => {
      const key = QUERY_KEYS.MUSIC.ANALYSIS('track-123');
      expect(key).toEqual(['music', 'analysis', 'track-123']);
    });

    it('should generate recommendations query key', () => {
      const key = QUERY_KEYS.MUSIC.RECOMMENDATIONS('track-123');
      expect(key).toEqual(['music', 'recommendations', 'track-123']);
    });

    it('should generate favorites query key with userId', () => {
      const key = QUERY_KEYS.MUSIC.FAVORITES('user-123');
      expect(key).toEqual(['music', 'favorites', 'user-123']);
    });

    it('should generate favorites query key without userId', () => {
      const key = QUERY_KEYS.MUSIC.FAVORITES();
      expect(key).toEqual(['music', 'favorites', undefined]);
    });

    it('should have static query keys', () => {
      expect(QUERY_KEYS.MUSIC.ANALYTICS).toEqual(['music', 'analytics']);
      expect(QUERY_KEYS.MUSIC.TRENDS).toEqual(['music', 'trends']);
      expect(QUERY_KEYS.ROBOT.STATUS).toEqual(['robot', 'status']);
      expect(QUERY_KEYS.ROBOT.COMMANDS).toEqual(['robot', 'commands']);
    });
  });

  describe('ROUTES', () => {
    it('should have correct route paths', () => {
      expect(ROUTES.HOME).toBe('/');
      expect(ROUTES.MUSIC).toBe('/music');
      expect(ROUTES.ROBOT).toBe('/robot');
      expect(ROUTES.NOT_FOUND).toBe('/404');
    });
  });

  describe('STORAGE_KEYS', () => {
    it('should have correct storage keys', () => {
      expect(STORAGE_KEYS.THEME).toBe('app-theme');
      expect(STORAGE_KEYS.USER_PREFERENCES).toBe('user-preferences');
      expect(STORAGE_KEYS.RECENT_SEARCHES).toBe('recent-searches');
      expect(STORAGE_KEYS.FAVORITES).toBe('favorites');
    });
  });

  describe('PAGINATION', () => {
    it('should have correct pagination defaults', () => {
      expect(PAGINATION.DEFAULT_PAGE_SIZE).toBe(20);
      expect(PAGINATION.MAX_PAGE_SIZE).toBe(100);
      expect(PAGINATION.DEFAULT_PAGE).toBe(1);
    });
  });

  describe('DEBOUNCE_DELAYS', () => {
    it('should have correct debounce delays', () => {
      expect(DEBOUNCE_DELAYS.SEARCH).toBe(500);
      expect(DEBOUNCE_DELAYS.INPUT).toBe(300);
      expect(DEBOUNCE_DELAYS.SCROLL).toBe(100);
    });
  });

  describe('TIMEOUTS', () => {
    it('should have correct timeout values', () => {
      expect(TIMEOUTS.API_REQUEST).toBe(30000);
      expect(TIMEOUTS.TOAST_DURATION).toBe(4000);
      expect(TIMEOUTS.RETRY_DELAY).toBe(1000);
    });
  });

  describe('VALIDATION_LIMITS', () => {
    it('should have correct validation limits', () => {
      expect(VALIDATION_LIMITS.SEARCH_QUERY_MIN_LENGTH).toBe(1);
      expect(VALIDATION_LIMITS.SEARCH_QUERY_MAX_LENGTH).toBe(100);
      expect(VALIDATION_LIMITS.TRACK_ID_MIN_LENGTH).toBe(1);
      expect(VALIDATION_LIMITS.MAX_TRACKS_COMPARE).toBe(10);
      expect(VALIDATION_LIMITS.MIN_TRACKS_COMPARE).toBe(2);
    });
  });

  describe('FEATURES', () => {
    it('should have feature flags', () => {
      expect(FEATURES).toHaveProperty('ENABLE_VOICE_COMMANDS');
      expect(FEATURES).toHaveProperty('ENABLE_OFFLINE_MODE');
      expect(FEATURES).toHaveProperty('ENABLE_ANALYTICS');
    });

    it('should read feature flags from environment', () => {
      // Feature flags are read from process.env
      expect(typeof FEATURES.ENABLE_VOICE_COMMANDS).toBe('boolean');
      expect(typeof FEATURES.ENABLE_OFFLINE_MODE).toBe('boolean');
      expect(typeof FEATURES.ENABLE_ANALYTICS).toBe('boolean');
    });
  });
});

