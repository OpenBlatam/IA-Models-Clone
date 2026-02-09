import { useEffect, useState } from 'react';
import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';

export interface DeepLink {
  url: string;
  path: string;
  queryParams: Record<string, string | string[] | undefined>;
}

export function useDeepLink() {
  const [initialUrl, setInitialUrl] = useState<string | null>(null);
  const [currentUrl, setCurrentUrl] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    // Get initial URL
    Linking.getInitialURL().then((url) => {
      if (url) {
        setInitialUrl(url);
        handleDeepLink(url);
      }
    });

    // Listen for deep links
    const subscription = Linking.addEventListener('url', (event) => {
      setCurrentUrl(event.url);
      handleDeepLink(event.url);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  function handleDeepLink(url: string) {
    const parsed = Linking.parse(url);
    const path = parsed.path || '';
    const queryParams = parsed.queryParams || {};

    // Handle routing based on path
    if (path.startsWith('/dashboard')) {
      router.push('/(tabs)/dashboard');
    } else if (path.startsWith('/jobs')) {
      router.push('/(tabs)/jobs');
    } else if (path.startsWith('/roadmap')) {
      router.push('/(tabs)/roadmap');
    } else if (path.startsWith('/profile')) {
      router.push('/(tabs)/profile');
    }
    // Add more routes as needed
  }

  return {
    initialUrl,
    currentUrl,
  };
}


