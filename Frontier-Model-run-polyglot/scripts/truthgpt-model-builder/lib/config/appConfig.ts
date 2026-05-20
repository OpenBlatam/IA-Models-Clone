/**
 * Configuración de la Aplicación
 * ===============================
 * 
 * Configuración centralizada y tipada
 */

import { APIClientOptions } from '../truthgpt-api-client-enhanced'

// ============================================================================
// CONFIGURACIÓN DE ENTORNO
// ============================================================================

export const ENV = {
  NODE_ENV: process.env.NODE_ENV || 'development',
  IS_DEVELOPMENT: process.env.NODE_ENV === 'development',
  IS_PRODUCTION: process.env.NODE_ENV === 'production',
  IS_TEST: process.env.NODE_ENV === 'test'
} as const

// ============================================================================
// CONFIGURACIÓN DE API
// ============================================================================

export const API_CONFIG: APIClientOptions = {
  baseUrl: process.env.NEXT_PUBLIC_TRUTHGPT_API_URL || 'http://localhost:8000',
  timeout: 30000,
  retries: 3,
  retryDelay: 1000,
  enableCache: true,
  cacheTTL: 60000,
  enableLogging: !ENV.IS_PRODUCTION,
  rateLimit: {
    maxRequests: 10,
    windowMs: 1000
  }
}

// ============================================================================
// CONFIGURACIÓN DE FEATURES
// ============================================================================

export const FEATURES = {
  ENABLE_ANALYTICS: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
  ENABLE_ERROR_TRACKING: process.env.NEXT_PUBLIC_ENABLE_ERROR_TRACKING === 'true',
  ENABLE_PERFORMANCE_MONITORING: process.env.NEXT_PUBLIC_ENABLE_PERFORMANCE_MONITORING === 'true',
  ENABLE_DEBUG_MODE: ENV.IS_DEVELOPMENT,
  ENABLE_HOT_RELOAD: ENV.IS_DEVELOPMENT
} as const

// ============================================================================
// CONFIGURACIÓN DE UI
// ============================================================================

export const UI_CONFIG = {
  THEME: {
    DEFAULT: 'dark' as const,
    AVAILABLE: ['light', 'dark', 'auto'] as const
  },
  LANGUAGE: {
    DEFAULT: 'es' as const,
    AVAILABLE: ['es', 'en'] as const
  },
  ANIMATIONS: {
    ENABLED: true,
    REDUCE_MOTION: false
  },
  ACCESSIBILITY: {
    ENABLE_SCREEN_READER: true,
    ENABLE_KEYBOARD_NAVIGATION: true,
    ENABLE_FOCUS_INDICATORS: true
  }
} as const

// ============================================================================
// CONFIGURACIÓN DE VALIDACIÓN
// ============================================================================

export const VALIDATION_CONFIG = {
  MODEL_NAME: {
    MIN_LENGTH: 3,
    MAX_LENGTH: 100,
    PATTERN: /^[a-zA-Z0-9\s-_]+$/
  },
  DESCRIPTION: {
    MIN_LENGTH: 10,
    MAX_LENGTH: 1000
  },
  EMAIL: {
    PATTERN: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  },
  URL: {
    PATTERN: /^https?:\/\/.+/
  }
} as const

// ============================================================================
// CONFIGURACIÓN DE PERFORMANCE
// ============================================================================

export const PERFORMANCE_CONFIG = {
  DEBOUNCE: {
    SEARCH: 500,
    INPUT: 300,
    RESIZE: 250
  },
  THROTTLE: {
    SCROLL: 100,
    RESIZE: 200
  },
  MEMOIZATION: {
    MAX_SIZE: 100,
    TTL: 60000
  },
  LAZY_LOADING: {
    THRESHOLD: 0.1,
    ROOT_MARGIN: '50px'
  }
} as const

// ============================================================================
// CONFIGURACIÓN DE CACHE
// ============================================================================

export const CACHE_CONFIG = {
  ENABLED: true,
  TTL: 60000, // 1 minuto
  MAX_SIZE: 100,
  STRATEGY: 'LRU' as const
} as const

// ============================================================================
// CONFIGURACIÓN DE LOGGING
// ============================================================================

export const LOGGING_CONFIG = {
  ENABLED: !ENV.IS_PRODUCTION,
  LEVEL: ENV.IS_PRODUCTION ? 'error' : 'debug',
  CONSOLE: ENV.IS_DEVELOPMENT,
  REMOTE: ENV.IS_PRODUCTION
} as const

// ============================================================================
// CONFIGURACIÓN DE ERROR HANDLING
// ============================================================================

export const ERROR_CONFIG = {
  SHOW_NOTIFICATIONS: true,
  LOG_TO_CONSOLE: true,
  SEND_TO_SERVICE: ENV.IS_PRODUCTION,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
} as const

// ============================================================================
// CONFIGURACIÓN COMPLETA
// ============================================================================

export const APP_CONFIG = {
  ENV,
  API: API_CONFIG,
  FEATURES,
  UI: UI_CONFIG,
  VALIDATION: VALIDATION_CONFIG,
  PERFORMANCE: PERFORMANCE_CONFIG,
  CACHE: CACHE_CONFIG,
  LOGGING: LOGGING_CONFIG,
  ERROR: ERROR_CONFIG
} as const

export type AppConfig = typeof APP_CONFIG







