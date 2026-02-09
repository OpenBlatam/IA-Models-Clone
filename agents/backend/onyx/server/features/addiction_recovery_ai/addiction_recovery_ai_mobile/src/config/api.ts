import Constants from 'expo-constants';

// API Configuration
const extra = Constants.expoConfig?.extra || {};

export const API_CONFIG = {
  BASE_URL: extra.apiBaseUrl || process.env.API_BASE_URL || 'http://localhost:8018',
  PREFIX: extra.apiPrefix || process.env.API_PREFIX || '/recovery',
  TIMEOUT: 30000,
};

export const getApiUrl = (endpoint: string): string => {
  const baseUrl = API_CONFIG.BASE_URL.replace(/\/$/, '');
  const prefix = API_CONFIG.PREFIX.replace(/^\/|\/$/g, '');
  const path = endpoint.replace(/^\/|\/$/g, '');
  return `${baseUrl}/${prefix}/${path}`;
};
