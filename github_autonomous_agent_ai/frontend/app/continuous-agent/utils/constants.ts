/**
 * General constants for Continuous Agent module
 * 
 * Centralized constants used across the module
 */

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  AGENTS: "/api/continuous-agent",
  AGENT: (id: string) => `/api/continuous-agent/${id}`,
  AGENT_LOGS: (id: string) => `/api/continuous-agent/${id}/logs`,
  STRIPE_CREDITS: "/api/continuous-agent/stripe/credits",
} as const;

/**
 * HTTP status codes
 */
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
} as const;

/**
 * Cache TTL values (in milliseconds)
 */
export const CACHE_TTL = {
  AGENTS_LIST: 10000, // 10 seconds
  SINGLE_AGENT: 5000, // 5 seconds
  AGENT_LOGS: 30000, // 30 seconds
  STRIPE_CREDITS: 60000, // 1 minute
} as const;

/**
 * Retry configuration defaults
 */
export const RETRY_CONFIG = {
  MAX_ATTEMPTS: 3,
  INITIAL_DELAY_MS: 1000,
  MAX_DELAY_MS: 30000,
  BACKOFF_MULTIPLIER: 2,
  JITTER: 0.1,
} as const;

/**
 * Debounce delays (in milliseconds)
 */
export const DEBOUNCE_DELAYS = {
  SEARCH: 300,
  FORM_INPUT: 500,
  RESIZE: 250,
  SCROLL: 100,
} as const;

/**
 * Breakpoints for responsive design
 */
export const BREAKPOINTS = {
  MOBILE: 768,
  TABLET: 1024,
  DESKTOP: 1280,
  WIDE: 1920,
} as const;

/**
 * Animation durations (in milliseconds)
 */
export const ANIMATION_DURATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
} as const;

/**
 * Z-index layers
 */
export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
} as const;




