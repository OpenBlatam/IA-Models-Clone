import { useEffect, useState } from 'react';
import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';

interface DeepLinkData {
  path: string;
  params?: Record<string, string>;
}

export function useDeepLinking() {
  const router = useRouter();
  const [initialUrl, setInitialUrl] = useState<string | null>(null);

  useEffect(() => {
    // Get initial URL if app was opened from a link
    Linking.getInitialURL().then((url) => {
      if (url) {
        setInitialUrl(url);
        handleDeepLink(url);
      }
    });

    // Listen for incoming links while app is running
    const subscription = Linking.addEventListener('url', (event) => {
      handleDeepLink(event.url);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  function handleDeepLink(url: string) {
    const { path, queryParams } = Linking.parse(url);
    
    if (!path) return;

    // Map deep link paths to app routes
    const routeMap: Record<string, string> = {
      'posts': '/(tabs)/posts',
      'post': `/(tabs)/posts/${queryParams?.id || ''}`,
      'calendar': '/(tabs)/calendar',
      'memes': '/(tabs)/memes',
      'meme': `/(tabs)/memes/${queryParams?.id || ''}`,
      'platforms': '/(tabs)/platforms',
      'analytics': '/(tabs)/analytics',
      'templates': '/(tabs)/templates',
      'template': `/(tabs)/templates/${queryParams?.id || ''}`,
      'dashboard': '/(tabs)/dashboard',
      'settings': '/(tabs)/settings',
    };

    const route = routeMap[path];
    if (route) {
      router.push(route as any);
    }
  }

  return { initialUrl };
}


