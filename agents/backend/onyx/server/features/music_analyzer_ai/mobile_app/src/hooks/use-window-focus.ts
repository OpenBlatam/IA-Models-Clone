import { useState, useEffect } from 'react';
import { AppState, AppStateStatus } from 'react-native';

/**
 * Hook to detect if app window is focused
 * Returns true when app is in foreground
 */
export function useWindowFocus(): boolean {
  const [isFocused, setIsFocused] = useState(() => AppState.currentState === 'active');

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState: AppStateStatus) => {
      setIsFocused(nextAppState === 'active');
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return isFocused;
}

