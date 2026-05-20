/**
 * Application constants.
 * Centralized constants for consistent usage across the application.
 */

/**
 * API endpoints configuration.
 */
export const API_ENDPOINTS = {
  MUSIC: {
    BASE: '/music',
    SEARCH: '/music/search',
    ANALYZE: '/music/analyze',
    TRACK: (id: string) => `/music/track/${id}`,
    COMPARE: '/music/compare',
    RECOMMENDATIONS: (id: string) => `/music/track/${id}/recommendations`,
    FAVORITES: '/music/favorites',
    HISTORY: '/music/history',
    ANALYTICS: '/music/analytics',
    HEALTH: '/music/health',
  },
  ROBOT: {
    BASE: '/robot',
    HEALTH: '/robot/health',
  },
} as const;

/**
 * Query keys for React Query.
 * Ensures consistent cache key usage across the application.
 */
export const QUERY_KEYS = {
  MUSIC: {
    SEARCH: (query: string) => ['music', 'search', query] as const,
    TRACK: (id: string) => ['music', 'track', id] as const,
    ANALYSIS: (id: string) => ['music', 'analysis', id] as const,
    RECOMMENDATIONS: (id: string) => ['music', 'recommendations', id] as const,
    FAVORITES: (userId?: string) => ['music', 'favorites', userId] as const,
    HISTORY: (userId?: string) => ['music', 'history', userId] as const,
    ANALYTICS: ['music', 'analytics'] as const,
    TRENDS: ['music', 'trends'] as const,
  },
  ROBOT: {
    STATUS: ['robot', 'status'] as const,
    COMMANDS: ['robot', 'commands'] as const,
  },
} as const;

/**
 * Application routes.
 */
export const ROUTES = {
  HOME: '/',
  MUSIC: '/music',
  ROBOT: '/robot',
  NOT_FOUND: '/404',
} as const;

/**
 * Local storage keys.
 */
export const STORAGE_KEYS = {
  THEME: 'app-theme',
  USER_PREFERENCES: 'user-preferences',
  RECENT_SEARCHES: 'recent-searches',
  FAVORITES: 'favorites',
} as const;

/**
 * Pagination defaults.
 */
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
  DEFAULT_PAGE: 1,
} as const;

/**
 * Debounce delays (in milliseconds).
 */
export const DEBOUNCE_DELAYS = {
  SEARCH: 500,
  INPUT: 300,
  SCROLL: 100,
} as const;

/**
 * Timeouts (in milliseconds).
 */
export const TIMEOUTS = {
  API_REQUEST: 30000,
  TOAST_DURATION: 4000,
  RETRY_DELAY: 1000,
} as const;

/**
 * Validation limits.
 */
export const VALIDATION_LIMITS = {
  SEARCH_QUERY_MIN_LENGTH: 1,
  SEARCH_QUERY_MAX_LENGTH: 100,
  TRACK_ID_MIN_LENGTH: 1,
  MAX_TRACKS_COMPARE: 10,
  MIN_TRACKS_COMPARE: 2,
} as const;

/**
 * Feature flags.
 */
export const FEATURES = {
  ENABLE_VOICE_COMMANDS: process.env.NEXT_PUBLIC_ENABLE_VOICE_COMMANDS === 'true',
  ENABLE_OFFLINE_MODE: process.env.NEXT_PUBLIC_ENABLE_OFFLINE_MODE === 'true',
  ENABLE_ANALYTICS: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
} as const;

