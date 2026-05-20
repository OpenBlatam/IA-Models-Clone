import { useEffect } from 'react';
import { useNavigation } from '@react-navigation/native';
import { metrics } from '../utils/metrics';

/**
 * Hook to track screen views automatically
 */
export const useScreenTracking = (screenName: string) => {
  const navigation = useNavigation();

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      metrics.trackScreenView(screenName);
    });

    return unsubscribe;
  }, [navigation, screenName]);
};

