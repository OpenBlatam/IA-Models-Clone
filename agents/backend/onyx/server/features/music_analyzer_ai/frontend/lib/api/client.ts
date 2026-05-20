/**
 * Axios client configuration and interceptors for API requests.
 * Enhanced with retry logic, health checks, and better error handling.
 */

import axios, {
  AxiosError,
  AxiosInstance,
  InternalAxiosRequestConfig,
  AxiosRequestConfig,
} from 'axios';
import { ApiError, NetworkError, getErrorMessage } from '../errors';
import { apiConfig } from '../config/app';
import { env } from '../config/env';

/**
 * Request retry configuration.
 */
interface RetryConfig {
  retries: number;
  retryDelay: number;
  retryCondition?: (error: AxiosError) => boolean;
}

/**
 * Default retry configuration.
 */
const defaultRetryConfig: RetryConfig = {
  retries: apiConfig.music.retries,
  retryDelay: 1000,
  retryCondition: (error: AxiosError) => {
    // Retry on network errors or 5xx server errors
    if (!error.response) {
      return true; // Network error
    }
    const status = error.response.status;
    return status >= 500 && status < 600; // Server errors
  },
};

/**
 * Sleep utility for retry delays.
 */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retry request with exponential backoff.
 */
async function retryRequest<T>(
  request: () => Promise<T>,
  config: RetryConfig = defaultRetryConfig
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= config.retries; attempt++) {
    try {
      return await request();
    } catch (error) {
      lastError = error as Error;

      // Don't retry if it's the last attempt
      if (attempt === config.retries) {
        break;
      }

      // Check if we should retry this error
      if (
        config.retryCondition &&
        error instanceof AxiosError &&
        !config.retryCondition(error)
      ) {
        break;
      }

      // Calculate delay with exponential backoff
      const delay = config.retryDelay * Math.pow(2, attempt);
      await sleep(delay);

      // Log retry attempt in development
      if (env.IS_DEVELOPMENT) {
        console.warn(
          `API request failed, retrying... (attempt ${attempt + 1}/${config.retries + 1})`
        );
      }
    }
  }

  throw lastError!;
}

/**
 * Creates a configured axios instance for the Music API.
 * @returns Configured axios instance
 */
export function createMusicApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: apiConfig.music.baseURL,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    timeout: apiConfig.music.timeout,
    withCredentials: false, // Set to true if using cookies/auth
  });

  // Request interceptor
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // Add request ID for tracking
      const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      config.metadata = { requestId, startTime: Date.now() };

      // Log request in development
      if (env.IS_DEVELOPMENT) {
        console.log(
          `[API Request] ${config.method?.toUpperCase()} ${config.url}`,
          {
            params: config.params,
            data: config.data,
            requestId,
          }
        );
      }

      // Add any auth tokens or headers here if needed
      // Example:
      // const token = getAuthToken();
      // if (token) {
      //   config.headers.Authorization = `Bearer ${token}`;
      // }

      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response) => {
      // Log response in development
      if (env.IS_DEVELOPMENT && response.config.metadata) {
        const duration = Date.now() - response.config.metadata.startTime;
        console.log(
          `[API Response] ${response.config.method?.toUpperCase()} ${response.config.url}`,
          {
            status: response.status,
            duration: `${duration}ms`,
            requestId: response.config.metadata.requestId,
          }
        );
      }

      return response;
    },
    async (error: AxiosError) => {
      // Log error in development
      if (env.IS_DEVELOPMENT) {
        console.error('[API Error]', {
          url: error.config?.url,
          method: error.config?.method,
          status: error.response?.status,
          message: error.message,
          requestId: error.config?.metadata?.requestId,
        });
      }

      // Handle network errors
      if (error.code === 'ECONNABORTED' || error.message === 'Network Error') {
        return Promise.reject(
          new NetworkError('Network connection failed. Please check your internet connection.')
        );
      }

      // Handle CORS errors
      if (error.code === 'ERR_NETWORK' && !error.response) {
        return Promise.reject(
          new NetworkError(
            'CORS error or network issue. Please check if the API server is running and accessible.'
          )
        );
      }

      // Handle API errors with response
      if (error.response) {
        const statusCode = error.response.status;
        const errorMessage =
          (error.response.data as { detail?: string })?.detail ||
          (error.response.data as { message?: string })?.message ||
          error.message ||
          'An API error occurred';

        // Handle specific status codes
        switch (statusCode) {
          case 401:
            return Promise.reject(
              new ApiError('Unauthorized. Please check your credentials.', statusCode, error.response.data)
            );
          case 403:
            return Promise.reject(
              new ApiError('Forbidden. You do not have permission to access this resource.', statusCode, error.response.data)
            );
          case 404:
            return Promise.reject(
              new ApiError('Resource not found.', statusCode, error.response.data)
            );
          case 429:
            return Promise.reject(
              new ApiError('Too many requests. Please try again later.', statusCode, error.response.data)
            );
          case 500:
          case 502:
          case 503:
          case 504:
            return Promise.reject(
              new ApiError('Server error. Please try again later.', statusCode, error.response.data)
            );
          default:
            return Promise.reject(
              new ApiError(errorMessage, statusCode, error.response.data)
            );
        }
      }

      return Promise.reject(new ApiError(getErrorMessage(error)));
    }
  );

  return client;
}

/**
 * Default music API client instance.
 */
export const musicApiClient = createMusicApiClient();

/**
 * Health check function to verify API connection.
 * @returns Promise resolving to health check response
 */
export async function checkApiHealth(): Promise<{
  status: 'healthy' | 'unhealthy';
  message: string;
  timestamp: number;
}> {
  try {
    const response = await musicApiClient.get('/health', {
      timeout: 5000, // Shorter timeout for health checks
    });

    return {
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      message: getErrorMessage(error),
      timestamp: Date.now(),
    };
  }
}

/**
 * Enhanced request function with retry logic.
 * @param requestFn - Function that returns a promise
 * @param retryConfig - Optional retry configuration
 * @returns Promise with retry logic applied
 */
export async function requestWithRetry<T>(
  requestFn: () => Promise<T>,
  retryConfig?: Partial<RetryConfig>
): Promise<T> {
  const config = { ...defaultRetryConfig, ...retryConfig };
  return retryRequest(requestFn, config);
}

// Extend AxiosRequestConfig to include metadata
declare module 'axios' {
  export interface AxiosRequestConfig {
    metadata?: {
      requestId: string;
      startTime: number;
    };
  }
}
