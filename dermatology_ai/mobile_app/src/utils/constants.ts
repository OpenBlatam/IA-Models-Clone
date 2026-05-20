/**
 * Constantes de la aplicación
 */

// Performance
export const PERFORMANCE_CONFIG = {
  FLATLIST: {
    WINDOW_SIZE: 10,
    MAX_TO_RENDER_PER_BATCH: 10,
    UPDATE_CELLS_BATCHING_PERIOD: 50,
    INITIAL_NUM_TO_RENDER: 10,
  },
  DEBOUNCE: {
    SEARCH: 300,
    INPUT: 500,
    SCROLL: 100,
  },
  THROTTLE: {
    SCROLL: 16,
    RESIZE: 100,
  },
} as const;

// Breakpoints responsivos
export const BREAKPOINTS = {
  XS: 0,
  SM: 576,
  MD: 768,
  LG: 992,
  XL: 1200,
} as const;

// Animation durations
export const ANIMATION_DURATION = {
  FAST: 200,
  NORMAL: 300,
  SLOW: 500,
} as const;

// Image optimization
export const IMAGE_CONFIG = {
  QUALITY: 0.8,
  MAX_WIDTH: 1920,
  MAX_HEIGHT: 1080,
  CACHE_SIZE: 100 * 1024 * 1024, // 100MB
} as const;
