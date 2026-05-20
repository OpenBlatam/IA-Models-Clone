import Constants from 'expo-constants';

export interface EnvironmentConfig {
  apiUrl: string;
  apiVersion: string;
  environment: 'development' | 'staging' | 'production';
  enableLogging: boolean;
  enableAnalytics: boolean;
  enableErrorTracking: boolean;
}

function getEnvironment(): EnvironmentConfig['environment'] {
  const env = Constants.expoConfig?.extra?.environment || process.env.NODE_ENV || 'development';
  if (env === 'prod' || env === 'production') return 'production';
  if (env === 'staging') return 'staging';
  return 'development';
}

export function getEnvironmentConfig(): EnvironmentConfig {
  const environment = getEnvironment();
  const apiUrl =
    Constants.expoConfig?.extra?.apiUrl ||
    process.env.EXPO_PUBLIC_API_URL ||
    (environment === 'production' ? 'https://api.example.com' : 'http://localhost:8030');

  return {
    apiUrl,
    apiVersion: 'v1',
    environment,
    enableLogging: environment !== 'production',
    enableAnalytics: environment === 'production',
    enableErrorTracking: environment === 'production',
  };
}

export const environment = getEnvironmentConfig();

