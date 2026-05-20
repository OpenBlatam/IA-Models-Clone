/**
 * Application constants
 */

// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010',
  WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8010',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
} as const;

// Robot Configuration
export const ROBOT_CONFIG = {
  MIN_POSITION: -10,
  MAX_POSITION: 10,
  DEFAULT_POSITION: { x: 0, y: 0, z: 0 },
  UPDATE_INTERVAL: 2000, // ms
  MOVEMENT_SPEED: 1.0, // m/s
} as const;

// UI Configuration
export const UI_CONFIG = {
  DEBOUNCE_DELAY: 300,
  THROTTLE_DELAY: 1000,
  ANIMATION_DURATION: 300,
  TOAST_DURATION: 4000,
  MODAL_ANIMATION_DURATION: 200,
} as const;

// Storage Keys
export const STORAGE_KEYS = {
  THEME: 'robot-theme',
  LANGUAGE: 'robot-language',
  ONBOARDING_COMPLETED: 'onboarding-completed',
  USER_PREFERENCES: 'user-preferences',
  RECENT_POSITIONS: 'recent-positions',
  FAVORITES: 'favorites',
} as const;

// Breakpoints
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// Animation Presets
export const ANIMATIONS = {
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
  },
  slideUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 },
  },
  slideDown: {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  },
  scale: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
  },
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  CONNECTION_FAILED: 'No se pudo conectar con el servidor',
  MOVEMENT_FAILED: 'Error al mover el robot',
  STOP_FAILED: 'Error al detener el robot',
  INVALID_POSITION: 'Posición inválida',
  WEBSOCKET_FAILED: 'Error de conexión WebSocket',
  UNKNOWN_ERROR: 'Ha ocurrido un error inesperado',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  MOVED: 'Robot movido exitosamente',
  STOPPED: 'Robot detenido',
  CONNECTED: 'Conectado al robot',
  SAVED: 'Configuración guardada',
} as const;



