import { useState, useEffect, useCallback } from 'react';
import { enhancedStorage } from '@/lib/utils/storage';

export function useLocalStorage<T>(
  key: string,
  initialValue: T,
  options?: { encrypt?: boolean; ttl?: number }
): [T, (value: T | ((val: T) => T)) => void, () => void] {
  // Get initial value from storage or use initial value
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      const item = enhancedStorage.get<T>(key, null, options?.encrypt);
      return item !== null ? item : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Set value function
  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      try {
        // Allow value to be a function so we have same API as useState
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        
        setStoredValue(valueToStore);
        
        if (typeof window !== 'undefined') {
          enhancedStorage.set(key, valueToStore, options);
        }
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key, storedValue, options]
  );

  // Remove value function
  const removeValue = useCallback(() => {
    try {
      setStoredValue(initialValue);
      if (typeof window !== 'undefined') {
        enhancedStorage.delete(key);
      }
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, initialValue]);

  // Listen for changes from other tabs/windows
  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue) {
        try {
          const item = enhancedStorage.get<T>(key, null, options?.encrypt);
          if (item !== null) {
            setStoredValue(item);
          }
        } catch (error) {
          console.error(`Error reading localStorage key "${key}" from storage event:`, error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [key, options?.encrypt]);

  return [storedValue, setValue, removeValue];
}
