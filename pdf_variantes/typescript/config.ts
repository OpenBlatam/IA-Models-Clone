/**
 * PDF Variantes API - TypeScript Configuration
 * Configuration for frontend integration
 */

export interface APIEnvironment {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
}

export const API_ENVIRONMENTS: Record<string, APIEnvironment> = {
  development: {
    baseURL: "http://localhost:8000",
    timeout: 60000, // 60 seconds for development
  },
  production: {
    baseURL: process.env.NEXT_PUBLIC_API_URL || "https://api.yourdomain.com",
    apiKey: process.env.NEXT_PUBLIC_API_KEY,
    timeout: 30000, // 30 seconds for production
  },
};

/**
 * Get the current environment configuration
 */
export function getAPIConfig(): APIEnvironment {
  const env = process.env.NODE_ENV || "development";
  return API_ENVIRONMENTS[env] || API_ENVIRONMENTS.development;
}

/**
 * Environment variables helpers
 */
export const API_CONFIG = {
  baseURL:
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.VITE_API_URL ||
    "http://localhost:8000",
  apiKey:
    process.env.NEXT_PUBLIC_API_KEY ||
    process.env.VITE_API_KEY ||
    undefined,
  timeout: 30000,
};






