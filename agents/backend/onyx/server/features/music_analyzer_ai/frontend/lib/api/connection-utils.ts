/**
 * API connection utilities.
 * Helper functions for managing API connections and handling connection issues.
 */

import { musicApiClient, checkApiHealth } from './client';
import { env } from '../config/env';
import { apiConfig } from '../config/app';

/**
 * Connection test result.
 */
export interface ConnectionTestResult {
  success: boolean;
  baseURL: string;
  message: string;
  responseTime?: number;
  error?: string;
}

/**
 * Tests the API connection with detailed information.
 * @returns Promise resolving to connection test result
 */
export async function testApiConnection(): Promise<ConnectionTestResult> {
  const startTime = Date.now();
  const baseURL = apiConfig.music.baseURL;

  try {
    const healthResult = await checkApiHealth();
    const responseTime = Date.now() - startTime;

    return {
      success: healthResult.status === 'healthy',
      baseURL,
      message: healthResult.message,
      responseTime,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    const errorMessage =
      error instanceof Error ? error.message : 'Unknown error';

    return {
      success: false,
      baseURL,
      message: 'Connection test failed',
      responseTime,
      error: errorMessage,
    };
  }
}

/**
 * Validates API configuration.
 * @returns Validation result with any issues found
 */
export function validateApiConfig(): {
  isValid: boolean;
  issues: string[];
} {
  const issues: string[] = [];

  // Check if base URL is set
  if (!apiConfig.music.baseURL || apiConfig.music.baseURL === '/music') {
    issues.push('Music API base URL is not properly configured');
  }

  // Check if URL is valid
  try {
    new URL(apiConfig.music.baseURL);
  } catch {
    issues.push('Music API base URL is not a valid URL');
  }

  // Check timeout
  if (apiConfig.music.timeout <= 0) {
    issues.push('API timeout must be greater than 0');
  }

  // Check retries
  if (apiConfig.music.retries < 0) {
    issues.push('API retries must be non-negative');
  }

  // Log configuration in development
  if (env.IS_DEVELOPMENT) {
    console.log('[API Config]', {
      baseURL: apiConfig.music.baseURL,
      timeout: apiConfig.music.timeout,
      retries: apiConfig.music.retries,
      environment: env.NODE_ENV,
    });
  }

  return {
    isValid: issues.length === 0,
    issues,
  };
}

/**
 * Gets API connection information.
 * @returns Connection information object
 */
export function getApiConnectionInfo(): {
  baseURL: string;
  timeout: number;
  retries: number;
  environment: string;
} {
  return {
    baseURL: apiConfig.music.baseURL,
    timeout: apiConfig.music.timeout,
    retries: apiConfig.music.retries,
    environment: env.NODE_ENV,
  };
}

