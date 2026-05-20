import { useState, useEffect, useRef } from 'react';
import { AppState, AppStateStatus } from 'react-native';

export function useAppState() {
  const appState = useRef(AppState.currentState);
  const [appStateVisible, setAppStateVisible] = useState<AppStateStatus>(appState.current);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
        // App has come to the foreground
      }

      appState.current = nextAppState;
      setAppStateVisible(appState.current);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return {
    appState: appStateVisible,
    isActive: appStateVisible === 'active',
    isBackground: appStateVisible === 'background',
    isInactive: appStateVisible === 'inactive',
  };
}


