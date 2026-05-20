import { useEffect, useRef } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { storeData, getData } from '../utils/storage-helpers';

/**
 * Hook to sync state with storage automatically
 * Persists state across app restarts
 */
export function useStorageSync<T>(
  key: string,
  value: T,
  setValue: (value: T) => void,
  encrypted = false
): void {
  const isInitialMount = useRef(true);
  const appState = useRef<AppStateStatus>(AppState.currentState);

  // Load from storage on mount
  useEffect(() => {
    async function loadFromStorage() {
      const stored = await getData<T>(key, encrypted);
      if (stored !== null) {
        setValue(stored);
      }
    }

    loadFromStorage();
  }, [key, encrypted, setValue]);

  // Save to storage when value changes (but not on initial mount)
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    async function saveToStorage() {
      await storeData(key, value, encrypted);
    }

    saveToStorage();
  }, [key, value, encrypted]);

  // Save to storage when app goes to background
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      if (
        appState.current.match(/active|foreground/) &&
        nextAppState === 'background'
      ) {
        storeData(key, value, encrypted);
      }
      appState.current = nextAppState;
    });

    return () => subscription.remove();
  }, [key, value, encrypted]);
}

