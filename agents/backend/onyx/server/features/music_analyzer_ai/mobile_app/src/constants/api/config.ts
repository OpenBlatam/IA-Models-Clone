import Constants from 'expo-constants';

export const API_BASE_URL =
  Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8010';

export const API_TIMEOUT = 30000;

export const API_RETRY_ATTEMPTS = 3;


