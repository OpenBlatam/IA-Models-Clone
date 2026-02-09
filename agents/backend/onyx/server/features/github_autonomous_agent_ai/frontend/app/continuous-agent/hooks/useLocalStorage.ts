/**
 * Custom hook for localStorage with type safety
 * 
 * Provides type-safe localStorage operations with React state sync
 */

import { useState, useEffect, useCallback } from "react";

/**
 * Options for useLocalStorage hook
 */
export interface UseLocalStorageOptions<T> {
  /** Default value if key doesn't exist */
  readonly defaultValue?: T;
  /** Serializer function */
  readonly serialize?: (value: T) => string;
  /** Deserializer function */
  readonly deserialize?: (value: string) => T;
}

/**
 * Return type for useLocalStorage hook
 */
export interface UseLocalStorageReturn<T> {
  /** Current value */
  readonly value: T;
  /** Set value */
  readonly setValue: (value: T | ((prev: T) => T)) => void;
  /** Remove value from storage */
  readonly removeValue: () => void;
  /** Whether value exists in storage */
  readonly hasValue: boolean;
}

/**
 * Custom hook for localStorage
 * 
 * @param key - Storage key
 * @param options - Storage options
 * @returns Value, setter, and storage utilities
 * 
 * @example
 * ```typescript
 * const { value, setValue } = useLocalStorage<AgentConfig>("agent-config", {
 *   defaultValue: { taskType: "custom", frequency: 3600 }
 * });
 * ```
 */
export function useLocalStorage<T>(
  key: string,
  options: UseLocalStorageOptions<T> = {}
): UseLocalStorageReturn<T> {
  const {
    defaultValue,
    serialize = JSON.stringify,
    deserialize = JSON.parse,
  } = options;

  // Get initial value from localStorage or use default
  const getStoredValue = useCallback((): T => {
    if (typeof window === "undefined") {
      return defaultValue as T;
    }

    try {
      const item = window.localStorage.getItem(key);
      if (item === null) {
        return defaultValue as T;
      }
      return deserialize(item);
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return defaultValue as T;
    }
  }, [key, defaultValue, deserialize]);

  const [value, setStoredValue] = useState<T>(getStoredValue);

  // Update state when localStorage changes (from other tabs/windows)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent): void => {
      if (e.key === key && e.newValue !== null) {
        try {
          setStoredValue(deserialize(e.newValue));
        } catch (error) {
          console.error(`Error parsing localStorage value for "${key}":`, error);
        }
      }
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [key, deserialize]);

  // Set value in both state and localStorage
  const setValue = useCallback(
    (newValue: T | ((prev: T) => T)): void => {
      try {
        const valueToStore =
          newValue instanceof Function ? newValue(value) : newValue;
        setStoredValue(valueToStore);

        if (typeof window !== "undefined") {
          window.localStorage.setItem(key, serialize(valueToStore));
        }
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key, value, serialize]
  );

  // Remove value from both state and localStorage
  const removeValue = useCallback((): void => {
    try {
      setStoredValue(defaultValue as T);
      if (typeof window !== "undefined") {
        window.localStorage.removeItem(key);
      }
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, defaultValue]);

  // Check if value exists
  const hasValue = useCallback((): boolean => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.localStorage.getItem(key) !== null;
  }, [key]);

  return {
    value,
    setValue,
    removeValue,
    hasValue: hasValue(),
  };
}




