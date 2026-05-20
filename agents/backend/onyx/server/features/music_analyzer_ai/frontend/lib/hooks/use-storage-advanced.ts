/**
 * Custom hook for advanced storage with expiration.
 * Provides storage with automatic expiration handling.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  setWithExpiration,
  getWithExpiration,
  clearExpired,
} from '../utils/storage-advanced';

/**
 * Options for useStorageAdvanced hook.
 */
export interface UseStorageAdvancedOptions<T> {
  expiration?: number; // milliseconds
  defaultValue?: T;
  onExpire?: () => void;
}

/**
 * Return type for useStorageAdvanced hook.
 */
export interface UseStorageAdvancedReturn<T> {
  value: T | null;
  setValue: (value: T, expiration?: number) => void;
  removeValue: () => void;
  isExpired: boolean;
}

/**
 * Custom hook for advanced storage with expiration.
 * Provides storage with automatic expiration.
 *
 * @param key - Storage key
 * @param options - Hook options
 * @returns Storage value and handlers
 */
export function useStorageAdvanced<T>(
  key: string,
  options: UseStorageAdvancedOptions<T> = {}
): UseStorageAdvancedReturn<T> {
  const { expiration, defaultValue, onExpire } = options;

  const [value, setValueState] = useState<T | null>(() => {
    if (typeof window === 'undefined') {
      return defaultValue || null;
    }

    clearExpired();
    return getWithExpiration<T>(key) || defaultValue || null;
  });

  const [isExpired, setIsExpired] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const checkExpiration = () => {
      const stored = getWithExpiration<T>(key);
      if (stored === null && value !== null) {
        setIsExpired(true);
        onExpire?.();
        setValueState(defaultValue || null);
      } else if (stored !== null) {
        setIsExpired(false);
        setValueState(stored);
      }
    };

    // Check expiration periodically
    const interval = setInterval(checkExpiration, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [key, value, defaultValue, onExpire]);

  const setValue = useCallback(
    (newValue: T, customExpiration?: number) => {
      const exp = customExpiration || expiration;
      if (exp) {
        setWithExpiration(key, newValue, exp);
      } else {
        localStorage.setItem(key, JSON.stringify(newValue));
      }
      setValueState(newValue);
      setIsExpired(false);
    },
    [key, expiration]
  );

  const removeValue = useCallback(() => {
    localStorage.removeItem(key);
    setValueState(defaultValue || null);
    setIsExpired(false);
  }, [key, defaultValue]);

  return {
    value,
    setValue,
    removeValue,
    isExpired,
  };
}

