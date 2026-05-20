import { useEffect } from 'react';
import { BackHandler, Platform } from 'react-native';

export function useBackHandler(handler: () => boolean) {
  useEffect(() => {
    if (Platform.OS !== 'android') {
      return;
    }

    const backHandler = BackHandler.addEventListener('hardwareBackPress', handler);

    return () => {
      backHandler.remove();
    };
  }, [handler]);
}


