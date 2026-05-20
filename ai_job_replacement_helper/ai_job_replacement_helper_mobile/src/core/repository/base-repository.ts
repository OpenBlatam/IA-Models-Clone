import axios, { AxiosInstance, AxiosError } from 'axios';
import { environment } from '../config/environment';
import * as SecureStore from 'react-native-encrypted-storage';
import { STORAGE_KEYS } from '@/constants/config';
import { NetworkError, AuthenticationError, ServerError, NotFoundError } from '../errors/app-error';

export abstract class BaseRepository {
  protected client: AxiosInstance;

  constructor(basePath: string = '') {
    this.client = axios.create({
      baseURL: `${environment.apiUrl}${basePath}`,
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
      async (config) => {
        const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
        if (sessionId && config.headers) {
          config.headers.Authorization = `Bearer ${sessionId}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await this.handleUnauthorized();
          return Promise.reject(new AuthenticationError());
        }

        const appError = this.mapAxiosError(error);
        return Promise.reject(appError);
      }
    );
  }

  private async handleUnauthorized(): Promise<void> {
    await SecureStore.removeItem(STORAGE_KEYS.SESSION_ID);
    await SecureStore.removeItem(STORAGE_KEYS.USER_ID);
    await SecureStore.removeItem(STORAGE_KEYS.USER_DATA);
  }

  private mapAxiosError(error: AxiosError): Error {
    if (!error.response) {
      return new NetworkError('Network error. Please check your connection.', error);
    }

    const status = error.response.status;
    const message = (error.response.data as any)?.detail || error.message;

    switch (status) {
      case 404:
        return new NotFoundError(message);
      case 401:
        return new AuthenticationError(message);
      case 400:
        return new ValidationError(message);
      case 500:
      case 502:
      case 503:
        return new ServerError(message, status);
      default:
        return new ServerError(message, status);
    }
  }

  protected async get<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  protected async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  protected async put<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  protected async delete<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  protected async patch<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }
}

