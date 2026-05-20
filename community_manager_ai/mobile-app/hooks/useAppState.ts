import { useState, useEffect, useRef } from 'react';
import { AppState, AppStateStatus } from 'react-native';

export function useAppState() {
  const appState = useRef(AppState.currentState);
  const [state, setState] = useState<AppStateStatus>(appState.current);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
        // App has come to the foreground
        console.log('App has come to the foreground');
      }

      if (appState.current === 'active' && nextAppState.match(/inactive|background/)) {
        // App has gone to the background
        console.log('App has gone to the background');
      }

      appState.current = nextAppState;
      setState(nextAppState);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return {
    state,
    isActive: state === 'active',
    isBackground: state === 'background',
    isInactive: state === 'inactive',
  };
}


