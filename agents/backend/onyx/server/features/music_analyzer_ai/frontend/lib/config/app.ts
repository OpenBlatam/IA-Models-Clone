/**
 * Application configuration.
 * Centralized application settings and configuration.
 */

import { env } from './env';

/**
 * Type definitions for configuration objects.
 */
export type UIConfig = typeof uiConfigInternal;
export type APIConfig = typeof apiConfigInternal;
export type PerformanceConfig = typeof performanceConfigInternal;
export type AppConfigType = typeof appConfig;

/**
 * UI configuration.
 */
const uiConfigInternal = {
  theme: {
    default: 'dark',
    storageKey: 'app-theme',
  },
  toast: {
    duration: 4000,
    position: 'top-right' as const,
    successDuration: 3000,
    errorDuration: 5000,
  },
  pagination: {
    defaultPageSize: 20,
    maxPageSize: 100,
  },
} as const;

/**
 * API configuration.
 */
const apiConfigInternal = {
  music: {
    baseURL: `${env.MUSIC_API_URL}/music`,
    timeout: 30000,
    retries: 2,
  },
  robot: {
    baseURL: `${env.ROBOT_API_URL}/robot`,
    timeout: 30000,
    retries: 2,
  },
} as const;

/**
 * Performance configuration.
 */
const performanceConfigInternal = {
  debounce: {
    search: 500,
    input: 300,
    scroll: 100,
  },
  cache: {
    staleTime: 60000, // 1 minute
    gcTime: 300000, // 5 minutes (formerly cacheTime)
  },
} as const;

/**
 * Application metadata configuration.
 * Unified configuration structure that includes all sub-configurations.
 */
export const appConfig = {
  name: 'Blatam Academy',
  description: 'Music Analyzer AI & Robot Movement AI Platform',
  version: '1.0.0',
  author: 'Blatam Academy',
  keywords: ['music analysis', 'AI', 'music analyzer', 'robot control'],
  url: env.IS_PRODUCTION
    ? 'https://blatam-academy.com'
    : 'http://localhost:3000',
  ui: uiConfigInternal,
  api: apiConfigInternal,
  performance: performanceConfigInternal,
} as const;

/**
 * API configuration.
 * Exported for backward compatibility and direct access.
 */
export const apiConfig = apiConfigInternal;

/**
 * UI configuration.
 * Exported for backward compatibility and direct access.
 */
export const uiConfig = uiConfigInternal;

/**
 * Performance configuration.
 * Exported for backward compatibility and direct access.
 */
export const performanceConfig = performanceConfigInternal;

