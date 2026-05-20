import * as Linking from 'expo-linking';

/**
 * Deep linking utilities
 */

export const createDeepLink = (path: string, params?: Record<string, string>): string => {
  const scheme = 'dermatologyai';
  let url = `${scheme}://${path}`;
  
  if (params) {
    const queryString = Object.entries(params)
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join('&');
    url += `?${queryString}`;
  }
  
  return url;
};

export const parseDeepLink = (url: string): { path: string; params: Record<string, string> } => {
  const parsed = Linking.parse(url);
  return {
    path: parsed.path || '',
    params: (parsed.queryParams || {}) as Record<string, string>,
  };
};

export const handleDeepLink = async (url: string) => {
  const { path, params } = parseDeepLink(url);
  
  // Handle different deep link paths
  switch (path) {
    case 'analysis':
      return { screen: 'Analysis', params };
    case 'recommendations':
      return { screen: 'Recommendations', params };
    case 'history':
      return { screen: 'History', params };
    default:
      return { screen: 'Home', params: {} };
  }
};

