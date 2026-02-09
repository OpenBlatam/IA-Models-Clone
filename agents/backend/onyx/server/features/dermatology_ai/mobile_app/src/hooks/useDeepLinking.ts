import { useEffect } from 'react';
import * as Linking from 'expo-linking';
import { useNavigation } from '@react-navigation/native';
import { handleDeepLink } from '../utils/deepLink';
import { RootStackParamList } from '../types';
import { StackNavigationProp } from '@react-navigation/stack';

type NavigationProp = StackNavigationProp<RootStackParamList>;

export const useDeepLinking = () => {
  const navigation = useNavigation<NavigationProp>();

  useEffect(() => {
    // Handle initial URL if app was opened via deep link
    Linking.getInitialURL().then((url) => {
      if (url) {
        handleDeepLink(url).then((route) => {
          navigation.navigate(route.screen as any, route.params);
        });
      }
    });

    // Handle deep links while app is running
    const subscription = Linking.addEventListener('url', (event) => {
      handleDeepLink(event.url).then((route) => {
        navigation.navigate(route.screen as any, route.params);
      });
    });

    return () => {
      subscription.remove();
    };
  }, [navigation]);
};

