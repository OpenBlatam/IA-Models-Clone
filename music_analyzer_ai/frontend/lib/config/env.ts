/**
 * Environment configuration.
 * Centralized environment variable access with validation and defaults.
 */

/**
 * Validates and returns environment variables.
 * Throws error if required variables are missing.
 */
function getEnvVar(key: string, defaultValue?: string): string {
  const value = process.env[key] || defaultValue;
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}

/**
 * Application environment configuration.
 */
export const env = {
  // API URLs
  MUSIC_API_URL: getEnvVar(
    'NEXT_PUBLIC_MUSIC_API_URL',
    'http://localhost:8010'
  ),
  ROBOT_API_URL: getEnvVar(
    'NEXT_PUBLIC_ROBOT_API_URL',
    'http://localhost:8010'
  ),

  // Environment
  NODE_ENV: process.env.NODE_ENV || 'development',
  IS_PRODUCTION: process.env.NODE_ENV === 'production',
  IS_DEVELOPMENT: process.env.NODE_ENV === 'development',

  // Feature flags
  ENABLE_VOICE_COMMANDS:
    process.env.NEXT_PUBLIC_ENABLE_VOICE_COMMANDS === 'true',
  ENABLE_OFFLINE_MODE:
    process.env.NEXT_PUBLIC_ENABLE_OFFLINE_MODE === 'true',
  ENABLE_ANALYTICS: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
} as const;

/**
 * Type-safe environment configuration.
 */
export type EnvConfig = typeof env;

