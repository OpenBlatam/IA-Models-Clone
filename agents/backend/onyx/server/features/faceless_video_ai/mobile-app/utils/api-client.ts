import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL, API_VERSION } from './config';
import type { ApiError, RateLimitInfo } from '@/types/api';

const TOKEN_KEY = 'auth_token';
const API_KEY_KEY = 'api_key';

class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;
  private apiKey: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
    this.loadCredentials();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      async (config) => {
        // Add auth token if available
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }

        // Add API key if available
        if (this.apiKey) {
          config.headers['X-API-Key'] = this.apiKey;
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        // Extract rate limit info from headers
        const rateLimitInfo: RateLimitInfo | null = response.headers['x-ratelimit-limit']
          ? {
              limit: parseInt(response.headers['x-ratelimit-limit'], 10),
              remaining: parseInt(response.headers['x-ratelimit-remaining'], 10),
              reset_at: parseInt(response.headers['x-ratelimit-reset'], 10),
            }
          : null;

        if (rateLimitInfo) {
          response.data._rateLimit = rateLimitInfo;
        }

        return response;
      },
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired, clear and redirect to login
          await this.clearCredentials();
        }

        const apiError: ApiError = {
          detail: error.response?.data?.detail || error.message || 'An error occurred',
          status_code: error.response?.status || 500,
          error_code: error.response?.data?.error_code,
        };

        return Promise.reject(apiError);
      }
    );
  }

  private async loadCredentials(): Promise<void> {
    try {
      this.token = await SecureStore.getItemAsync(TOKEN_KEY);
      this.apiKey = await SecureStore.getItemAsync(API_KEY_KEY);
    } catch (error) {
      console.error('Failed to load credentials:', error);
    }
  }

  async setToken(token: string): Promise<void> {
    this.token = token;
    try {
      await SecureStore.setItemAsync(TOKEN_KEY, token);
    } catch (error) {
      console.error('Failed to save token:', error);
    }
  }

  async setApiKey(apiKey: string): Promise<void> {
    this.apiKey = apiKey;
    try {
      await SecureStore.setItemAsync(API_KEY_KEY, apiKey);
    } catch (error) {
      console.error('Failed to save API key:', error);
    }
  }

  async clearCredentials(): Promise<void> {
    this.token = null;
    this.apiKey = null;
    try {
      await SecureStore.deleteItemAsync(TOKEN_KEY);
      await SecureStore.deleteItemAsync(API_KEY_KEY);
    } catch (error) {
      console.error('Failed to clear credentials:', error);
    }
  }

  getToken(): string | null {
    return this.token;
  }

  getApiKey(): string | null {
    return this.apiKey;
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  async upload<T>(
    url: string,
    file: { uri: string; type: string; name: string },
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', {
      uri: file.uri,
      type: file.type,
      name: file.name,
    } as unknown as Blob);

    const response = await this.client.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  async download(
    url: string,
    onProgress?: (progress: number) => void
  ): Promise<{ uri: string; headers: Record<string, string> }> {
    const response = await this.client.get(url, {
      responseType: 'blob',
      onDownloadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });

    // In React Native, we need to handle blob differently
    // This is a simplified version - you may need to adjust based on your needs
    return {
      uri: url,
      headers: response.headers as Record<string, string>,
    };
  }
}

export const apiClient = new ApiClient();


