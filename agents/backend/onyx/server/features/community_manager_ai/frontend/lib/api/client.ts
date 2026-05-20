/**
 * API Client
 * Centralized Axios instance with interceptors for error handling and authentication
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { env } from '@/lib/config/env';
import { handleApiError } from '@/lib/errors/handler';
import { AppError } from '@/lib/errors/types';

/**
 * Creates and configures the API client instance
 * @returns Configured Axios instance
 */
export const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: env.apiUrl,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 30000, // 30 seconds
  });

  // Request interceptor for adding auth tokens
  client.interceptors.request.use(
    (config) => {
      // Add auth token if available
      // This can be extended to get token from session or store
      return config;
    },
    (error) => {
      return Promise.reject(handleApiError(error));
    }
  );

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response: AxiosResponse) => response,
    (error) => {
      return Promise.reject(handleApiError(error));
    }
  );

  return client;
};

/**
 * Default API client instance
 */
export const apiClient = createApiClient();

/**
 * Type-safe API request wrapper
 * @param request - The API request function
 * @returns Promise with typed response data
 * @throws {AppError} If the request fails
 */
export const apiRequest = async <T>(
  request: () => Promise<AxiosResponse<T>>
): Promise<T> => {
  try {
    const response = await request();
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Type-safe GET request helper
 * @param url - The endpoint URL
 * @param config - Optional Axios request config
 * @returns Promise with typed response data
 */
export const apiGet = async <T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> => {
  return apiRequest(() => apiClient.get<T>(url, config));
};

/**
 * Type-safe POST request helper
 * @param url - The endpoint URL
 * @param data - The request body
 * @param config - Optional Axios request config
 * @returns Promise with typed response data
 */
export const apiPost = async <T>(
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> => {
  return apiRequest(() => apiClient.post<T>(url, data, config));
};

/**
 * Type-safe PUT request helper
 * @param url - The endpoint URL
 * @param data - The request body
 * @param config - Optional Axios request config
 * @returns Promise with typed response data
 */
export const apiPut = async <T>(
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> => {
  return apiRequest(() => apiClient.put<T>(url, data, config));
};

/**
 * Type-safe DELETE request helper
 * @param url - The endpoint URL
 * @param config - Optional Axios request config
 * @returns Promise with typed response data
 */
export const apiDelete = async <T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> => {
  return apiRequest(() => apiClient.delete<T>(url, config));
};


