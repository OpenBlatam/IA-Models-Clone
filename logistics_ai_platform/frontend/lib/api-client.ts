import axios, { AxiosInstance, AxiosError } from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030';

const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  client.interceptors.request.use(async (config) => {
    if (typeof window !== 'undefined') {
      const { getSession } = await import('next-auth/react');
      const session = await getSession();
      if (session?.accessToken) {
        config.headers.Authorization = `Bearer ${session.accessToken}`;
      }
    }
    return config;
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      if (error.response?.status === 401 && typeof window !== 'undefined') {
        const locale = window.location.pathname.split('/')[1] || 'en';
        window.location.href = `/${locale}/auth/signin`;
      }
      return Promise.reject(error);
    }
  );

  return client;
};

export const apiClient = createApiClient();

