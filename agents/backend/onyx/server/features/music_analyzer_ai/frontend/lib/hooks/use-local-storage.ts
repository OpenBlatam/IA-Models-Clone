/**
 * Custom hook for localStorage with type safety and SSR support.
 * Provides reactive localStorage access with automatic synchronization.
 */

import { useState, useEffect, useCallback } from 'react';

/**
 * Options for useLocalStorage hook.
 */
export interface UseLocalStorageOptions<T> {
  serializer?: {
    read: (value: string) => T;
    write: (value: T) => string;
  };
  onError?: (error: Error) => void;
}

/**
 * Return type for useLocalStorage hook.
 */
export interface UseLocalStorageReturn<T> {
  value: T;
  setValue: (value: T | ((prev: T) => T)) => void;
  removeValue: () => void;
  isLoaded: boolean;
}

/**
 * Custom hook for localStorage with type safety.
 * Automatically syncs with localStorage and handles SSR.
 *
 * @param key - Storage key
 * @param initialValue - Initial value if key doesn't exist
 * @param options - Hook options
 * @returns LocalStorage state and handlers
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T,
  options: UseLocalStorageOptions<T> = {}
): UseLocalStorageReturn<T> {
  const { serializer, onError } = options;

  // State to hold our value
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      const item = window.localStorage.getItem(key);
      if (item === null) {
        return initialValue;
      }

      if (serializer) {
        return serializer.read(item);
      }

      return JSON.parse(item) as T;
    } catch (error) {
      const errorInstance =
        error instanceof Error ? error : new Error(String(error));
      onError?.(errorInstance);
      return initialValue;
    }
  });

  const [isLoaded, setIsLoaded] = useState(false);

  // Load from storage on mount
  useEffect(() => {
    if (typeof window === 'undefined') {
      setIsLoaded(true);
      return;
    }

    try {
      const item = window.localStorage.getItem(key);
      if (item !== null) {
        const parsed = serializer
          ? serializer.read(item)
          : (JSON.parse(item) as T);
        setStoredValue(parsed);
      }
    } catch (error) {
      const errorInstance =
        error instanceof Error ? error : new Error(String(error));
      onError?.(errorInstance);
    } finally {
      setIsLoaded(true);
    }
  }, [key, serializer, onError]);

  // Listen for storage changes from other tabs/windows
  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const parsed = serializer
            ? serializer.read(e.newValue)
            : (JSON.parse(e.newValue) as T);
          setStoredValue(parsed);
        } catch (error) {
          const errorInstance =
            error instanceof Error ? error : new Error(String(error));
          onError?.(errorInstance);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [key, serializer, onError]);

  /**
   * Sets the value in state and localStorage.
   */
  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      try {
        // Allow value to be a function so we have the same API as useState
        const valueToStore =
          value instanceof Function ? value(storedValue) : value;

        // Save state
        setStoredValue(valueToStore);

        // Save to local storage
        if (typeof window !== 'undefined') {
          const serialized = serializer
            ? serializer.write(valueToStore)
            : JSON.stringify(valueToStore);
          window.localStorage.setItem(key, serialized);
        }
      } catch (error) {
        const errorInstance =
          error instanceof Error ? error : new Error(String(error));
        onError?.(errorInstance);
      }
    },
    [key, serializer, storedValue, onError]
  );

  /**
   * Removes the value from state and localStorage.
   */
  const removeValue = useCallback(() => {
    try {
      setStoredValue(initialValue);
      if (typeof window !== 'undefined') {
        window.localStorage.removeItem(key);
      }
    } catch (error) {
      const errorInstance =
        error instanceof Error ? error : new Error(String(error));
      onError?.(errorInstance);
    }
  }, [key, initialValue, onError]);

  return {
    value: storedValue,
    setValue,
    removeValue,
    isLoaded,
  };
}
