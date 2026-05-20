/**
 * Environment configuration with validation
 * Centralizes all environment variables with type safety
 */

/**
 * Validates and returns the API URL from environment variables
 * @throws {Error} If API URL is not configured
 */
export const getApiUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_API_URL;
  
  if (!url) {
    if (process.env.NODE_ENV === 'production') {
      throw new Error('NEXT_PUBLIC_API_URL is required in production');
    }
    return 'http://localhost:8000';
  }
  
  return url;
};

/**
 * Application environment configuration
 */
export const env = {
  apiUrl: getApiUrl(),
  nodeEnv: process.env.NODE_ENV || 'development',
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
  isTest: process.env.NODE_ENV === 'test',
} as const;


