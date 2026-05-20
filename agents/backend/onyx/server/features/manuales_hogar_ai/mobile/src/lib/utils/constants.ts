/**
 * Constants
 * ========
 * Application-wide constants
 */

export const APP_NAME = 'Manuales Hogar AI';
export const APP_VERSION = '1.0.0';

// API
export const API_TIMEOUT = 30000; // 30 seconds
export const MAX_RETRIES = 3;

// Images
export const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
export const MAX_IMAGES = 5;
export const IMAGE_QUALITY = 0.8;

// Manuals
export const MAX_MANUAL_DESCRIPTION_LENGTH = 2000;
export const MIN_MANUAL_DESCRIPTION_LENGTH = 10;

// Cache
export const CACHE_STALE_TIME = 5 * 60 * 1000; // 5 minutes
export const CACHE_GC_TIME = 10 * 60 * 1000; // 10 minutes

// Pagination
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;

// Subscription limits
export const FREE_TIER_LIMIT = 5;
export const BASIC_TIER_LIMIT = 20;
export const PREMIUM_TIER_LIMIT = Infinity;

// Debounce delays
export const SEARCH_DEBOUNCE = 500;
export const INPUT_DEBOUNCE = 300;

// Toast durations
export const TOAST_DURATION_SHORT = 2000;
export const TOAST_DURATION_MEDIUM = 3000;
export const TOAST_DURATION_LONG = 4000;



