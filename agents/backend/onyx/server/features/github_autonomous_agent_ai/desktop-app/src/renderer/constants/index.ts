/**
 * Application Constants
 */

// API Configuration
export const API_CONFIG = {
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
} as const;

// Refresh Intervals (in milliseconds)
export const REFRESH_INTERVALS = {
  AGENTS_LIST: 5000,
  TASKS_LIST: 10000,
  STATUS: 3000,
} as const;

// Task Status
export const TASK_STATUS = {
  PENDING: 'pending',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;

// Agent Status
export const AGENT_STATUS = {
  ACTIVE: 'active',
  INACTIVE: 'inactive',
  PAUSED: 'paused',
} as const;

// Local Storage Keys
export const STORAGE_KEYS = {
  API_KEY: 'api_key',
  GITHUB_TOKEN: 'github_access_token',
  GITHUB_USER: 'github_user',
  SELECTED_MODEL: 'selected_ai_model',
  SELECTED_REPOSITORY: 'selected_repository',
} as const;


