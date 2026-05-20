import { useEffect, useState, useCallback } from 'react';

export const useLocalStorageSync = <T,>(
  key: string,
  initialValue: T,
  options: {
    syncAcrossTabs?: boolean;
    serializer?: (value: T) => string;
    deserializer?: (value: string) => T;
  } = {}
) => {
  const { syncAcrossTabs = true, serializer = JSON.stringify, deserializer = JSON.parse } = options;

  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      const item = window.localStorage.getItem(key);
      return item ? deserializer(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      try {
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        setStoredValue(valueToStore);

        if (typeof window !== 'undefined') {
          window.localStorage.setItem(key, serializer(valueToStore));
        }
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key, serializer, storedValue]
  );

  useEffect(() => {
    if (!syncAcrossTabs || typeof window === 'undefined') {
      return;
    }

    const handleStorageChange = (e: StorageEvent): void => {
      if (e.key === key && e.newValue) {
        try {
          setStoredValue(deserializer(e.newValue));
        } catch (error) {
          console.error(`Error parsing localStorage value for key "${key}":`, error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [key, syncAcrossTabs, deserializer]);

  return [storedValue, setValue] as const;
};



