import Constants from 'expo-constants';

/**
 * App configuration constants
 */

export const APP_CONFIG = {
  NAME: 'Community Manager AI',
  VERSION: Constants.expoConfig?.version || '1.0.0',
  API_URL: Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8000',
  ENVIRONMENT: __DEV__ ? 'development' : 'production',
} as const;

/**
 * API Configuration
 */
export const API_CONFIG = {
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
} as const;

/**
 * Storage Keys
 */
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  THEME: 'theme',
  LANGUAGE: 'language',
  USER_PREFERENCES: 'user_preferences',
} as const;

/**
 * Platform IDs
 */
export const PLATFORMS = {
  FACEBOOK: 'facebook',
  INSTAGRAM: 'instagram',
  TWITTER: 'twitter',
  LINKEDIN: 'linkedin',
  TIKTOK: 'tiktok',
  YOUTUBE: 'youtube',
} as const;

/**
 * Post Status
 */
export const POST_STATUS = {
  SCHEDULED: 'scheduled',
  PUBLISHED: 'published',
  CANCELLED: 'cancelled',
} as const;

/**
 * Validation Limits
 */
export const VALIDATION_LIMITS = {
  POST_CONTENT_MIN: 1,
  POST_CONTENT_MAX: 5000,
  MEME_CAPTION_MAX: 500,
  TEMPLATE_NAME_MIN: 1,
  TEMPLATE_NAME_MAX: 100,
  TEMPLATE_CONTENT_MIN: 1,
  TEMPLATE_CONTENT_MAX: 10000,
  TAG_MAX_LENGTH: 50,
  MAX_TAGS: 10,
} as const;

/**
 * Animation Durations
 */
export const ANIMATION_DURATIONS = {
  FAST: 200,
  NORMAL: 300,
  SLOW: 500,
} as const;

/**
 * Debounce Delays
 */
export const DEBOUNCE_DELAYS = {
  SEARCH: 500,
  INPUT: 300,
  SCROLL: 100,
} as const;


