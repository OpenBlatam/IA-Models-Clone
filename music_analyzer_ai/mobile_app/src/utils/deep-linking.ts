import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';

interface DeepLinkConfig {
  [key: string]: (params: Record<string, string>) => void;
}

/**
 * Deep linking configuration
 * Maps URL paths to navigation handlers
 */
const deepLinkHandlers: DeepLinkConfig = {
  '/search': (params) => {
    // Handle search deep link
    const query = params.q || params.query || '';
    // Navigate to search with query
  },
  '/track/:id': (params) => {
    // Handle track deep link
    const trackId = params.id;
    // Navigate to track analysis
  },
  '/analysis/:trackId': (params) => {
    // Handle analysis deep link
    const trackId = params.trackId;
    // Navigate to analysis screen
  },
};

/**
 * Initialize deep linking
 */
export function initializeDeepLinking(router: ReturnType<typeof useRouter>): () => void {
  const handleUrl = (event: { url: string }) => {
    const { path, queryParams } = Linking.parse(event.url);

    if (!path) {
      return;
    }

    // Find matching handler
    for (const [pattern, handler] of Object.entries(deepLinkHandlers)) {
      const regex = new RegExp(
        pattern.replace(/:[^/]+/g, '([^/]+)').replace(/\//g, '\\/')
      );
      const match = path.match(regex);

      if (match) {
        const params: Record<string, string> = { ...queryParams };
        const paramNames = pattern.match(/:[^/]+/g) || [];

        paramNames.forEach((paramName, index) => {
          const key = paramName.substring(1);
          params[key] = match[index + 1] || '';
        });

        handler(params);
        return;
      }
    }
  };

  // Handle initial URL
  Linking.getInitialURL().then((url) => {
    if (url) {
      handleUrl({ url });
    }
  });

  // Handle URL changes
  const subscription = Linking.addEventListener('url', handleUrl);

  return () => {
    subscription.remove();
  };
}

/**
 * Generate deep link URL
 */
export function generateDeepLink(path: string, params?: Record<string, string>): string {
  const baseUrl = Linking.createURL(path);
  
  if (!params || Object.keys(params).length === 0) {
    return baseUrl;
  }

  const queryString = Object.entries(params)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
    .join('&');

  return `${baseUrl}?${queryString}`;
}

