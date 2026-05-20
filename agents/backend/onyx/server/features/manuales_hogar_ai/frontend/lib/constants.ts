export const DEFAULT_USER_ID = 'user-1';

export const PAGINATION = {
  DEFAULT_LIMIT: 20,
  DEFAULT_OFFSET: 0,
  MAX_LIMIT: 100,
} as const;

export const FILE_UPLOAD = {
  MAX_IMAGES: 5,
  MAX_SIZE_MB: 10,
  ACCEPTED_TYPES: ['image/jpeg', 'image/png', 'image/webp', 'image/gif'],
} as const;

export const SEARCH = {
  DEFAULT_LIMIT: 20,
  DEFAULT_THRESHOLD: 0.5,
  DEBOUNCE_MS: 300,
} as const;

export const ANALYTICS = {
  DEFAULT_DAYS: 30,
  INTERVALS: ['day', 'week'] as const,
} as const;

export const QUERY_CONFIG = {
  STALE_TIME: 60 * 1000,
  GC_TIME: 5 * 60 * 1000,
  RETRY: 1,
  RETRY_DELAY: 1000,
} as const;

export const TOAST_CONFIG = {
  DURATION: 4000,
  SUCCESS_DURATION: 3000,
  ERROR_DURATION: 5000,
} as const;

