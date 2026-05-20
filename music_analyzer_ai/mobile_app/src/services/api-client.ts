import axios, {
  AxiosInstance,
  AxiosError,
  InternalAxiosRequestConfig,
  AxiosResponse,
  CancelTokenSource,
} from 'axios';
import { API_BASE_URL, API_TIMEOUT, API_RETRY_ATTEMPTS } from '../constants/api';
import { ApiError } from '../types/api';
import { apiLogger } from '../utils/logger/logger';
import { logError } from '../utils/error-handler';
import { retryWithNetworkCheck } from '../utils/retry-utils';
import { requestQueue } from '../utils/api/request-queue';
import { loadSecureData } from '../utils/storage';

interface RequestConfig extends InternalAxiosRequestConfig {
  _retry?: boolean;
  _skipRetry?: boolean;
}

const AUTH_TOKEN_KEY = '@music_analyzer:auth_token';

class ApiClient {
  private client: AxiosInstance;
  private cancelTokens: Map<string, CancelTokenSource> = new Map();

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      async (config: RequestConfig) => {
        apiLogger.debug(`Request: ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data,
        });

        const token = await this.getAuthToken();
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        return config;
      },
      (error) => {
        apiLogger.error('Request error', error);
        return Promise.reject(error);
      }
    );

    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        apiLogger.debug(
          `Response: ${response.config.method?.toUpperCase()} ${response.config.url}`,
          {
            status: response.status,
            data: response.data,
          }
        );
        return response;
      },
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await this.clearAuthToken();
        }

        const config = error.config as RequestConfig;

        if (!config || config._skipRetry) {
          return this.handleError(error);
        }

        if (this.shouldRetry(error) && !config._retry) {
          config._retry = true;
          return this.retryRequest(config);
        }

        return this.handleError(error);
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    if (!error.response) {
      return true;
    }

    const status = error.response.status;
    return status >= 500 || status === 408 || status === 429;
  }

  private async retryRequest(config: RequestConfig): Promise<AxiosResponse> {
    try {
      return await retryWithNetworkCheck(
        () => this.client.request(config),
        {
          maxRetries: API_RETRY_ATTEMPTS,
          delay: 1000,
          backoff: 'exponential',
          onRetry: (attempt, err) => {
            apiLogger.warn(`Retrying request (attempt ${attempt})`, {
              url: config.url,
              error: err.message,
            });
          },
        }
      );
    } catch (error) {
      return Promise.reject(error);
    }
  }

  private handleError(error: AxiosError): Promise<never> {
    const apiError: ApiError = {
      detail:
        (error.response?.data as { detail?: string })?.detail ||
        error.message ||
        'An unexpected error occurred',
      status_code: error.response?.status || 500,
    };

    apiLogger.error('API Error', {
      url: error.config?.url,
      method: error.config?.method,
      status: apiError.status_code,
      detail: apiError.detail,
    });

    logError(apiError, 'API');

    return Promise.reject(apiError);
  }

  private async getAuthToken(): Promise<string | null> {
    try {
      return await loadSecureData(AUTH_TOKEN_KEY);
    } catch (error) {
      apiLogger.warn('Failed to load auth token', { error });
      return null;
    }
  }

  private getRequestKey(method: string, url: string, params?: unknown): string {
    const paramsStr = params ? JSON.stringify(params) : '';
    return `${method}-${url}-${paramsStr}`;
  }

  private async executeRequest<T>(
    method: 'get' | 'post' | 'put' | 'delete' | 'patch',
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    const requestKey = this.getRequestKey(method, url, config?.params || data);
    const cachedRequest = requestQueue.get<T>(requestKey);

    if (cachedRequest) {
      apiLogger.debug('Using cached request', { url, method });
      const response = await cachedRequest;
      return response.data;
    }

    const cancelTokenSource = axios.CancelToken.source();
    this.cancelTokens.set(requestKey, cancelTokenSource);

    const requestConfig: InternalAxiosRequestConfig = {
      ...config,
      cancelToken: cancelTokenSource.token,
    };

    let promise: Promise<AxiosResponse<T>>;

    try {
      switch (method) {
        case 'get':
          promise = this.client.get<T>(url, requestConfig);
          break;
        case 'post':
          promise = this.client.post<T>(url, data, requestConfig);
          break;
        case 'put':
          promise = this.client.put<T>(url, data, requestConfig);
          break;
        case 'delete':
          promise = this.client.delete<T>(url, requestConfig);
          break;
        case 'patch':
          promise = this.client.patch<T>(url, data, requestConfig);
          break;
        default:
          throw new Error(`Unsupported HTTP method: ${method}`);
      }

      requestQueue.set(requestKey, promise);

      const response = await promise;
      this.cancelTokens.delete(requestKey);
      return response.data;
    } catch (error) {
      this.cancelTokens.delete(requestKey);
      if (axios.isCancel(error)) {
        apiLogger.debug('Request cancelled', { url, method });
        throw new Error('Request was cancelled');
      }
      throw error;
    }
  }

  async get<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    return this.executeRequest<T>('get', url, undefined, config);
  }

  async post<T>(
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    return this.executeRequest<T>('post', url, data, config);
  }

  async put<T>(
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    return this.executeRequest<T>('put', url, data, config);
  }

  async delete<T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> {
    return this.executeRequest<T>('delete', url, undefined, config);
  }

  async patch<T>(
    url: string,
    data?: unknown,
    config?: InternalAxiosRequestConfig
  ): Promise<T> {
    return this.executeRequest<T>('patch', url, data, config);
  }

  async setAuthToken(token: string | null): Promise<void> {
    try {
      if (token) {
        const { saveSecureData } = await import('../utils/storage');
        await saveSecureData(AUTH_TOKEN_KEY, token);
        this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      } else {
        await this.clearAuthToken();
      }
    } catch (error) {
      apiLogger.error('Failed to set auth token', { error });
      throw error;
    }
  }

  async clearAuthToken(): Promise<void> {
    try {
      const { loadSecureData } = await import('../utils/storage');
      const EncryptedStorage = (await import('react-native-encrypted-storage')).default;
      await EncryptedStorage.removeItem(AUTH_TOKEN_KEY);
      delete this.client.defaults.headers.common['Authorization'];
    } catch (error) {
      apiLogger.error('Failed to clear auth token', { error });
    }
  }

  cancelRequest(requestKey: string): void {
    const cancelToken = this.cancelTokens.get(requestKey);
    if (cancelToken) {
      cancelToken.cancel('Request cancelled');
      this.cancelTokens.delete(requestKey);
    }
  }

  cancelAllRequests(): void {
    this.cancelTokens.forEach((cancelToken) => {
      cancelToken.cancel('All requests cancelled');
    });
    this.cancelTokens.clear();
  }
}

export const apiClient = new ApiClient();

