import axios, { AxiosError, AxiosInstance } from 'axios';
import { API_CONFIG } from '@/config/api.config';

export const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_CONFIG.BASE_URL,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: API_CONFIG.TIMEOUT,
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      const message =
        error.response?.data &&
        typeof error.response.data === 'object' &&
        'message' in error.response.data
          ? (error.response.data as { message: string }).message
          : error.message || 'An error occurred';

      return Promise.reject(new Error(message));
    }
  );

  return client;
};

export const apiClient = createApiClient();
