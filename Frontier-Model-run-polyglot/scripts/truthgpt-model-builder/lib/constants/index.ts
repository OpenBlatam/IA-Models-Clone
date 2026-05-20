/**
 * Constantes de la Aplicación
 * ===========================
 * 
 * Constantes centralizadas para toda la aplicación
 */

// ============================================================================
// API
// ============================================================================

export const API_BASE_URL = process.env.NEXT_PUBLIC_TRUTHGPT_API_URL || 'http://localhost:8000'
export const API_TIMEOUT = 30000 // 30 segundos
export const API_RETRIES = 3
export const API_RETRY_DELAY = 1000 // 1 segundo

// ============================================================================
// CACHE
// ============================================================================

export const CACHE_TTL = 60000 // 1 minuto
export const CACHE_MAX_SIZE = 100
export const CACHE_ENABLED = true

// ============================================================================
// RATE LIMITING
// ============================================================================

export const RATE_LIMIT_MAX_REQUESTS = 10
export const RATE_LIMIT_WINDOW_MS = 1000 // 1 segundo

// ============================================================================
// DEBOUNCE & THROTTLE
// ============================================================================

export const DEBOUNCE_DELAY = 500 // 500ms
export const THROTTLE_DELAY = 1000 // 1 segundo

// ============================================================================
// POLLING
// ============================================================================

export const POLL_INTERVAL = 2000 // 2 segundos
export const POLL_MAX_ATTEMPTS = 100
export const POLL_TIMEOUT = 300000 // 5 minutos

// ============================================================================
// VALIDATION
// ============================================================================

export const MIN_MODEL_NAME_LENGTH = 3
export const MAX_MODEL_NAME_LENGTH = 100
export const MIN_DESCRIPTION_LENGTH = 10
export const MAX_DESCRIPTION_LENGTH = 1000

// ============================================================================
// UI
// ============================================================================

export const TOAST_DURATION = 5000 // 5 segundos
export const TOAST_DURATION_ERROR = 7000 // 7 segundos
export const TOAST_DURATION_SUCCESS = 3000 // 3 segundos

export const MODAL_ANIMATION_DURATION = 300 // ms
export const LOADING_SPINNER_SIZE = {
  sm: 16,
  md: 24,
  lg: 32,
  xl: 48
} as const

// ============================================================================
// BREAKPOINTS
// ============================================================================

export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536
} as const

// ============================================================================
// COLORS
// ============================================================================

export const COLORS = {
  primary: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a'
  },
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d'
  },
  error: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d'
  },
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f'
  }
} as const

// ============================================================================
// MODEL STATUS
// ============================================================================

export const MODEL_STATUS = {
  IDLE: 'idle',
  VALIDATING: 'validating',
  CREATING: 'creating',
  COMPILING: 'compiling',
  TRAINING: 'training',
  EVALUATING: 'evaluating',
  PREDICTING: 'predicting',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
  PAUSED: 'paused'
} as const

export const MODEL_STATUS_LABELS: Record<string, string> = {
  [MODEL_STATUS.IDLE]: 'Inactivo',
  [MODEL_STATUS.VALIDATING]: 'Validando',
  [MODEL_STATUS.CREATING]: 'Creando',
  [MODEL_STATUS.COMPILING]: 'Compilando',
  [MODEL_STATUS.TRAINING]: 'Entrenando',
  [MODEL_STATUS.EVALUATING]: 'Evaluando',
  [MODEL_STATUS.PREDICTING]: 'Prediciendo',
  [MODEL_STATUS.COMPLETED]: 'Completado',
  [MODEL_STATUS.FAILED]: 'Fallido',
  [MODEL_STATUS.CANCELLED]: 'Cancelado',
  [MODEL_STATUS.PAUSED]: 'Pausado'
} as const

// ============================================================================
// MODEL TYPES
// ============================================================================

export const MODEL_TYPES = {
  CLASSIFICATION: 'classification',
  REGRESSION: 'regression',
  NLP: 'nlp',
  VISION: 'vision',
  TIME_SERIES: 'time-series',
  GENERATIVE: 'generative',
  CUSTOM: 'custom'
} as const

// ============================================================================
// OPTIMIZERS
// ============================================================================

export const OPTIMIZERS = {
  ADAM: 'adam',
  SGD: 'sgd',
  RMSPROP: 'rmsprop',
  ADAGRAD: 'adagrad',
  ADAMW: 'adamw',
  NADAM: 'nadam'
} as const

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

export const LOSS_FUNCTIONS = {
  SPARSE_CATEGORICAL_CROSSENTROPY: 'sparse_categorical_crossentropy',
  CATEGORICAL_CROSSENTROPY: 'categorical_crossentropy',
  BINARY_CROSSENTROPY: 'binary_crossentropy',
  MEAN_SQUARED_ERROR: 'mean_squared_error',
  MEAN_ABSOLUTE_ERROR: 'mean_absolute_error',
  HUBER: 'huber',
  COSINE_SIMILARITY: 'cosine_similarity'
} as const

// ============================================================================
// METRICS
// ============================================================================

export const METRICS = {
  ACCURACY: 'accuracy',
  PRECISION: 'precision',
  RECALL: 'recall',
  F1_SCORE: 'f1_score',
  AUC: 'auc',
  MAE: 'mae',
  MSE: 'mse',
  RMSE: 'rmse',
  R2_SCORE: 'r2_score'
} as const

// ============================================================================
// STORAGE KEYS
// ============================================================================

export const STORAGE_KEYS = {
  USER_PREFERENCES: 'user-preferences',
  MODEL_HISTORY: 'model-history',
  RECENT_MODELS: 'recent-models',
  FAVORITES: 'favorites',
  SETTINGS: 'settings',
  THEME: 'theme',
  LANGUAGE: 'language'
} as const

// ============================================================================
// ERROR CODES
// ============================================================================

export const ERROR_CODES = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  API_ERROR: 'API_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
  RATE_LIMIT_ERROR: 'RATE_LIMIT_ERROR',
  AUTH_ERROR: 'AUTH_ERROR',
  NOT_FOUND: 'NOT_FOUND',
  SERVER_ERROR: 'SERVER_ERROR'
} as const

// ============================================================================
// REGEX PATTERNS
// ============================================================================

export const REGEX_PATTERNS = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  URL: /^https?:\/\/.+/,
  PHONE: /^\+?[\d\s-()]+$/,
  ALPHANUMERIC: /^[a-zA-Z0-9]+$/,
  NUMERIC: /^\d+$/,
  SLUG: /^[a-z0-9-]+$/
} as const

// ============================================================================
// ANIMATION DURATIONS
// ============================================================================

export const ANIMATION_DURATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
  VERY_SLOW: 1000
} as const

// ============================================================================
// Z-INDEX LAYERS
// ============================================================================

export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
  TOAST: 1080
} as const







