// App Constants
export const APP_NAME = 'Addiction Recovery AI';
export const APP_VERSION = '1.0.0';

// API Constants
export const API_TIMEOUT = 30000;
export const API_RETRY_ATTEMPTS = 3;
export const API_RETRY_DELAY = 1000;

// Cache Constants
export const CACHE_TTL = {
  SHORT: 60 * 1000, // 1 minute
  MEDIUM: 5 * 60 * 1000, // 5 minutes
  LONG: 30 * 60 * 1000, // 30 minutes
  VERY_LONG: 24 * 60 * 60 * 1000, // 24 hours
};

// Performance Constants
export const DEBOUNCE_DELAY = 300;
export const THROTTLE_DELAY = 1000;
export const ANIMATION_DURATION = 300;

// List Constants
export const LIST_CONFIG = {
  ESTIMATED_ITEM_SIZE: 50,
  INITIAL_NUM_TO_RENDER: 10,
  MAX_TO_RENDER_PER_BATCH: 10,
  WINDOW_SIZE: 10,
  UPDATE_CELLS_BATCHING_PERIOD: 50,
};

// Image Constants
export const IMAGE_CONFIG = {
  MAX_WIDTH: 1200,
  MAX_HEIGHT: 1200,
  QUALITY: 0.8,
  CACHE_TTL: CACHE_TTL.VERY_LONG,
};

// Validation Constants
export const VALIDATION = {
  MIN_PASSWORD_LENGTH: 8,
  MAX_INPUT_LENGTH: 500,
  MAX_TEXTAREA_LENGTH: 2000,
};

// Storage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  USER_DATA: 'user_data',
  THEME: 'theme',
  LANGUAGE: 'language',
  ONBOARDING_COMPLETE: 'onboarding_complete',
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  UNAUTHORIZED: 'Session expired. Please login again.',
  NOT_FOUND: 'Resource not found.',
  SERVER_ERROR: 'Server error. Please try again later.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  UNKNOWN_ERROR: 'An unexpected error occurred.',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Welcome back!',
  REGISTER_SUCCESS: 'Account created successfully!',
  UPDATE_SUCCESS: 'Updated successfully!',
  DELETE_SUCCESS: 'Deleted successfully!',
} as const;
