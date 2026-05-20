/**
 * Storage utility functions.
 * Provides type-safe localStorage and sessionStorage operations.
 */

/**
 * Storage interface for abstraction.
 */
interface StorageInterface {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
}

/**
 * Checks if storage is available.
 * @param storage - Storage to check
 * @returns True if storage is available
 */
function isStorageAvailable(storage: StorageInterface): boolean {
  try {
    const testKey = '__storage_test__';
    storage.setItem(testKey, 'test');
    storage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Gets an item from storage with type safety.
 * @param storage - Storage to use
 * @param key - Storage key
 * @param defaultValue - Default value if not found
 * @returns Parsed value or default
 */
function getStorageItem<T>(
  storage: StorageInterface,
  key: string,
  defaultValue: T
): T {
  if (!isStorageAvailable(storage)) {
    return defaultValue;
  }

  try {
    const item = storage.getItem(key);
    if (item === null) {
      return defaultValue;
    }
    return JSON.parse(item) as T;
  } catch {
    return defaultValue;
  }
}

/**
 * Sets an item in storage with type safety.
 * @param storage - Storage to use
 * @param key - Storage key
 * @param value - Value to store
 */
function setStorageItem<T>(storage: StorageInterface, key: string, value: T): void {
  if (!isStorageAvailable(storage)) {
    return;
  }

  try {
    storage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Error setting storage item ${key}:`, error);
  }
}

/**
 * Removes an item from storage.
 * @param storage - Storage to use
 * @param key - Storage key
 */
function removeStorageItem(storage: StorageInterface, key: string): void {
  if (!isStorageAvailable(storage)) {
    return;
  }

  try {
    storage.removeItem(key);
  } catch (error) {
    console.error(`Error removing storage item ${key}:`, error);
  }
}

/**
 * LocalStorage utilities with type safety.
 */
export const localStorage = {
  /**
   * Gets an item from localStorage.
   */
  getItem: <T>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
      return defaultValue;
    }
    return getStorageItem(window.localStorage, key, defaultValue);
  },

  /**
   * Sets an item in localStorage.
   */
  setItem: <T>(key: string, value: T): void => {
    if (typeof window === 'undefined') {
      return;
    }
    setStorageItem(window.localStorage, key, value);
  },

  /**
   * Removes an item from localStorage.
   */
  removeItem: (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    removeStorageItem(window.localStorage, key);
  },

  /**
   * Clears all items from localStorage.
   */
  clear: (): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },
};

/**
 * SessionStorage utilities with type safety.
 */
export const sessionStorage = {
  /**
   * Gets an item from sessionStorage.
   */
  getItem: <T>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
      return defaultValue;
    }
    return getStorageItem(window.sessionStorage, key, defaultValue);
  },

  /**
   * Sets an item in sessionStorage.
   */
  setItem: <T>(key: string, value: T): void => {
    if (typeof window === 'undefined') {
      return;
    }
    setStorageItem(window.sessionStorage, key, value);
  },

  /**
   * Removes an item from sessionStorage.
   */
  removeItem: (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    removeStorageItem(window.sessionStorage, key);
  },

  /**
   * Clears all items from sessionStorage.
   */
  clear: (): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.sessionStorage.clear();
    } catch (error) {
      console.error('Error clearing sessionStorage:', error);
    }
  },
};

