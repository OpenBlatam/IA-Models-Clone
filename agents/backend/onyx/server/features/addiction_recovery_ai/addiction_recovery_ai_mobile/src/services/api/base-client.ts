import axios, { AxiosInstance, AxiosError } from 'axios';
import { secureStorage } from '@/utils/secure-storage';
import { API_CONFIG, getApiUrl } from '@/config/api';

export class BaseApiClient {
  protected client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      async (config) => {
        const token = await secureStorage.getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await secureStorage.removeToken();
        }
        return Promise.reject(error);
      }
    );
  }

  protected getUrl(endpoint: string): string {
    return getApiUrl(endpoint);
  }

  async setToken(token: string): Promise<void> {
    await secureStorage.setToken(token);
  }

  async getToken(): Promise<string | null> {
    return await secureStorage.getToken();
  }

  async clearToken(): Promise<void> {
    await secureStorage.removeToken();
  }
}

