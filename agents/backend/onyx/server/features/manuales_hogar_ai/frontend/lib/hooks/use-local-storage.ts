import { useState, useEffect, useCallback } from 'react';

interface UseLocalStorageReturn<T> {
  value: T | null;
  setValue: (value: T | null) => void;
  removeValue: () => void;
  isLoading: boolean;
}

export const useLocalStorage = <T,>(
  key: string,
  initialValue: T | null = null
): UseLocalStorageReturn<T> => {
  const [value, setValueState] = useState<T | null>(initialValue);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        setValueState(JSON.parse(item));
      }
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
    } finally {
      setIsLoading(false);
    }
  }, [key]);

  const setValue = useCallback(
    (newValue: T | null): void => {
      try {
        if (newValue === null) {
          window.localStorage.removeItem(key);
        } else {
          window.localStorage.setItem(key, JSON.stringify(newValue));
        }
        setValueState(newValue);
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key]
  );

  const removeValue = useCallback((): void => {
    try {
      window.localStorage.removeItem(key);
      setValueState(null);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key]);

  return {
    value,
    setValue,
    removeValue,
    isLoading,
  };
};

