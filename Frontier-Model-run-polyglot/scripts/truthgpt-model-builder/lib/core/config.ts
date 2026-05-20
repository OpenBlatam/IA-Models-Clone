/**
 * Core configuration and constants
 */

export const CONFIG = {
  // Paths
  TRUTHGPT_PATH: '../TruthGPT-main',
  GENERATED_MODELS_DIR: 'generated_models',
  
  // Cache
  SPEC_CACHE_TTL: 5 * 60 * 1000, // 5 minutes
  
  // Model creation
  MIN_DESCRIPTION_LENGTH: 10,
  MAX_DESCRIPTION_LENGTH: 1000,
  
  // Retry settings
  RETRY_MAX_ATTEMPTS: 3,
  RETRY_INITIAL_DELAY: 500,
  RETRY_MAX_DELAY: 10000,
  
  // Performance
  MODEL_CREATION_TIMEOUT: 20000, // 20 seconds
  
  // Validation
  MIN_LAYERS: 1,
  MAX_LAYERS: 10,
  MIN_LAYER_SIZE: 32,
  MAX_LAYER_SIZE: 10000,
  MIN_LEARNING_RATE: 0.00001,
  MAX_LEARNING_RATE: 0.1,
  MIN_BATCH_SIZE: 1,
  MAX_BATCH_SIZE: 1024,
  MIN_EPOCHS: 1,
  MAX_EPOCHS: 1000,
  MIN_DROPOUT: 0,
  MAX_DROPOUT: 0.5,
  
  // Default values
  DEFAULT_LEARNING_RATE: 0.001,
  DEFAULT_BATCH_SIZE: 32,
  DEFAULT_EPOCHS: 10,
  DEFAULT_DROPOUT: 0.2,
  
  // Auto-save
  AUTO_SAVE_INTERVAL: 2000, // 2 seconds
  DRAFT_MAX_AGE_MINUTES: 30,
} as const

export type ConfigKey = keyof typeof CONFIG


