/**
 * API helper utilities
 */

import { handleApiError, handleNetworkError } from './error-handler';
import { retryWithBackoff } from './helpers';
import { API_CONFIG } from './constants';

export interface ApiRequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: any;
  retries?: number;
  timeout?: number;
}

/**
 * Make API request with error handling and retry
 */
export async function apiRequest<T>(
  endpoint: string,
  options: ApiRequestOptions = {}
): Promise<T> {
  const {
    method = 'GET',
    headers = {},
    body,
    retries = API_CONFIG.RETRY_ATTEMPTS,
    timeout = API_CONFIG.TIMEOUT,
  } = options;

  const url = endpoint.startsWith('http')
    ? endpoint
    : `${API_CONFIG.BASE_URL}${endpoint}`;

  const requestOptions: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  };

  try {
    const response = await retryWithBackoff(
      async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
          const res = await fetch(url, {
            ...requestOptions,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          return res.json();
        } catch (error) {
          clearTimeout(timeoutId);
          throw error;
        }
      },
      retries,
      API_CONFIG.RETRY_DELAY
    );

    return response as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      handleNetworkError(new Error('Request timeout'));
    } else {
      handleApiError(error);
    }
    throw error;
  }
}

/**
 * GET request helper
 */
export function apiGet<T>(endpoint: string, options?: Omit<ApiRequestOptions, 'method' | 'body'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'GET' });
}

/**
 * POST request helper
 */
export function apiPost<T>(endpoint: string, body?: any, options?: Omit<ApiRequestOptions, 'method'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'POST', body });
}

/**
 * PUT request helper
 */
export function apiPut<T>(endpoint: string, body?: any, options?: Omit<ApiRequestOptions, 'method'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'PUT', body });
}

/**
 * DELETE request helper
 */
export function apiDelete<T>(endpoint: string, options?: Omit<ApiRequestOptions, 'method' | 'body'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'DELETE' });
}

/**
 * PATCH request helper
 */
export function apiPatch<T>(endpoint: string, body?: any, options?: Omit<ApiRequestOptions, 'method'>): Promise<T> {
  return apiRequest<T>(endpoint, { ...options, method: 'PATCH', body });
}



