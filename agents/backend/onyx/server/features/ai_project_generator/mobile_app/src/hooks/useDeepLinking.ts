import { useEffect } from 'react';
import * as Linking from 'expo-linking';
import { useNavigation } from '@react-navigation/native';

export const useDeepLinking = () => {
  const navigation = useNavigation();

  useEffect(() => {
    const handleInitialURL = async () => {
      const initialUrl = await Linking.getInitialURL();
      if (initialUrl) {
        handleURL(initialUrl);
      }
    };

    const handleURL = (url: string) => {
      const { path, queryParams } = Linking.parse(url);
      
      if (path === 'project' && queryParams?.id) {
        navigation.navigate('ProjectDetail' as never, { projectId: queryParams.id } as never);
      } else if (path === 'projects') {
        navigation.navigate('Projects' as never);
      } else if (path === 'generate') {
        navigation.navigate('Generate' as never);
      } else if (path === 'home') {
        navigation.navigate('Home' as never);
      }
    };

    handleInitialURL();

    const subscription = Linking.addEventListener('url', (event) => {
      handleURL(event.url);
    });

    return () => {
      subscription.remove();
    };
  }, [navigation]);
};

export const createDeepLink = (path: string, params?: Record<string, string>): string => {
  const baseUrl = 'ai-project-generator://';
  const queryString = params
    ? '?' + Object.entries(params).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')
    : '';
  return `${baseUrl}${path}${queryString}`;
};

