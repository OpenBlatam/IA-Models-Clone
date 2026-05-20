import { useState, useEffect } from 'react';
import { AppState, type AppStateStatus } from 'react-native';

/**
 * Hook to monitor app state (foreground/background)
 * Useful for pausing/resuming operations
 */
export function useAppState(): AppStateStatus {
  const [appState, setAppState] = useState<AppStateStatus>(
    AppState.currentState
  );

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      setAppState(nextAppState);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return appState;
}

/**
 * Hook to check if app is in foreground
 */
export function useIsForeground(): boolean {
  const appState = useAppState();
  return appState === 'active';
}

/**
 * Hook that runs callback when app comes to foreground
 */
export function useOnForeground(callback: () => void): void {
  const appState = useAppState();

  useEffect(() => {
    if (appState === 'active') {
      callback();
    }
  }, [appState, callback]);
}

/**
 * Hook that runs callback when app goes to background
 */
export function useOnBackground(callback: () => void): void {
  const appState = useAppState();

  useEffect(() => {
    if (appState === 'background' || appState === 'inactive') {
      callback();
    }
  }, [appState, callback]);
}

