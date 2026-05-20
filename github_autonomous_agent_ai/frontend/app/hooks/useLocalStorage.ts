import { useState } from 'react';

export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] {
  // Estado para almacenar nuestro valor
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }
    try {
      const item = window.localStorage.getItem(key);
      if (!item) {
        return initialValue;
      }
      // Try to parse as JSON first
      try {
        return JSON.parse(item);
      } catch (parseError) {
        // If parsing fails, check if it's a plain string that might have been stored incorrectly
        // This handles cases where localStorage was set directly without JSON.stringify
        if (item.trim().startsWith('"') && item.trim().endsWith('"')) {
          // It looks like a JSON string but parsing failed, return initialValue
          console.warn(`Invalid JSON in localStorage key "${key}", resetting to initial value`);
          return initialValue;
        }
        // If it's a plain string and the initialValue is also a string, use the stored value
        // Otherwise, return initialValue
        if (typeof initialValue === 'string') {
          return item as T;
        }
        console.warn(`Invalid JSON in localStorage key "${key}", resetting to initial value`);
        return initialValue;
      }
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Función para actualizar el valor
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      // Permitir que value sea una función para que tengamos la misma API que useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  };

  return [storedValue, setValue];
}

