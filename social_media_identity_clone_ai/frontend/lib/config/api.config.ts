export const apiConfig = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  apiVersion: 'v1',
  timeout: 30000,
  retries: 3,
} as const;

export const getApiBaseUrl = (): string => {
  return `${apiConfig.baseURL}/api/${apiConfig.apiVersion}`;
};



