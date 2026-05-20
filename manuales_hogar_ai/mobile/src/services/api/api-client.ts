/**
 * API Client
 * ==========
 * Centralized API client using Axios with interceptors
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import Constants from 'expo-constants';
import * as SecureStore from 'expo-secure-store';

const API_BASE_URL =
  Constants.expoConfig?.extra?.apiUrl ||
  process.env.EXPO_PUBLIC_API_URL ||
  'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        // Add auth token if available
        const token = await SecureStore.getItemAsync('auth_token');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error: AxiosError) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response) {
          // Handle specific error codes
          switch (error.response.status) {
            case 401:
              // Unauthorized - clear token and redirect to login
              await SecureStore.deleteItemAsync('auth_token');
              break;
            case 403:
              // Forbidden
              break;
            case 404:
              // Not found
              break;
            case 429:
              // Rate limited
              break;
            case 500:
              // Server error
              break;
          }
        } else if (error.request) {
          // Network error
          console.error('Network error:', error.message);
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  async postFormData<T>(
    url: string,
    formData: FormData,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    const response = await this.client.post<T>(url, formData, {
      ...config,
      headers: {
        ...config?.headers,
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  getBaseURL(): string {
    return API_BASE_URL;
  }
}

export const apiClient = new ApiClient();




