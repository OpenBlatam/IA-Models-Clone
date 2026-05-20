// API Constants
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const API_TIMEOUT = 30000; // 30 seconds

// Storage Keys
export const STORAGE_KEYS = {
  SETTINGS: 'settings',
  FAVORITES: 'favorites',
  SEARCH_HISTORY: 'search_history',
  NOTIFICATIONS: 'notifications',
  AUTOSAVE: 'autosave',
  TUTORIAL_COMPLETED: 'tutorial_completed',
  WELCOME_SHOWN: 'welcome_shown',
  CONTEXTUAL_HELP: 'contextual_help',
  NOTIFICATION_SOUNDS: 'notification_sounds',
  THEME: 'theme',
  BOOKMARKS: 'bookmarks',
  COLOR_TAGS: 'color_tags',
  DOCUMENT_HISTORY: 'document_history',
} as const;

// Pagination
export const DEFAULT_PAGE_SIZE = 20;
export const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];

// Debounce Delays
export const DEBOUNCE_DELAYS = {
  SEARCH: 300,
  INPUT: 500,
  RESIZE: 150,
} as const;

// Animation Durations
export const ANIMATION_DURATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
} as const;

// Breakpoints
export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// Max Values
export const MAX_VALUES = {
  QUERY_LENGTH: 5000,
  SEARCH_HISTORY: 50,
  NOTIFICATIONS: 100,
  DOCUMENT_VERSIONS: 20,
} as const;

