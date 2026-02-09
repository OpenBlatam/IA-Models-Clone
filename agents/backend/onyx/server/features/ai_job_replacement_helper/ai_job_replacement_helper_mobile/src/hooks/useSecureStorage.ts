import { useState, useEffect, useCallback } from 'react';
import * as SecureStore from 'react-native-encrypted-storage';

export function useSecureStorage<T>(key: string, initialValue: T | null = null) {
  const [storedValue, setStoredValue] = useState<T | null>(initialValue);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadValue() {
      try {
        const item = await SecureStore.getItem(key);
        if (item) {
          setStoredValue(JSON.parse(item) as T);
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to load from storage'));
      } finally {
        setIsLoading(false);
      }
    }

    loadValue();
  }, [key]);

  const setValue = useCallback(
    async (value: T | null) => {
      try {
        if (value === null) {
          await SecureStore.removeItem(key);
          setStoredValue(null);
        } else {
          await SecureStore.setItem(key, JSON.stringify(value));
          setStoredValue(value);
        }
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to save to storage'));
      }
    },
    [key]
  );

  const removeValue = useCallback(async () => {
    try {
      await SecureStore.removeItem(key);
      setStoredValue(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to remove from storage'));
    }
  }, [key]);

  return { value: storedValue, setValue, removeValue, isLoading, error };
}


