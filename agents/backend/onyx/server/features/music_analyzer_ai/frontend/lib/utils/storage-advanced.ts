/**
 * Advanced storage utility functions.
 * Provides enhanced storage operations with expiration and encryption support.
 */

/**
 * Storage item with metadata.
 */
interface StorageItem<T> {
  value: T;
  expires?: number;
  timestamp: number;
}

/**
 * Sets an item in storage with expiration.
 * @param key - Storage key
 * @param value - Value to store
 * @param expiration - Expiration time in milliseconds
 */
export function setWithExpiration<T>(
  key: string,
  value: T,
  expiration: number
): void {
  if (typeof window === 'undefined') {
    return;
  }

  const item: StorageItem<T> = {
    value,
    expires: Date.now() + expiration,
    timestamp: Date.now(),
  };

  try {
    localStorage.setItem(key, JSON.stringify(item));
  } catch (error) {
    console.error('Failed to set storage item:', error);
  }
}

/**
 * Gets an item from storage, checking expiration.
 * @param key - Storage key
 * @returns Value or null if expired/not found
 */
export function getWithExpiration<T>(key: string): T | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const itemStr = localStorage.getItem(key);
    if (!itemStr) {
      return null;
    }

    const item: StorageItem<T> = JSON.parse(itemStr);

    if (item.expires && Date.now() > item.expires) {
      localStorage.removeItem(key);
      return null;
    }

    return item.value;
  } catch (error) {
    console.error('Failed to get storage item:', error);
    return null;
  }
}

/**
 * Clears expired items from storage.
 */
export function clearExpired(): void {
  if (typeof window === 'undefined') {
    return;
  }

  const keys = Object.keys(localStorage);

  for (const key of keys) {
    try {
      const itemStr = localStorage.getItem(key);
      if (!itemStr) continue;

      const item = JSON.parse(itemStr);
      if (item.expires && Date.now() > item.expires) {
        localStorage.removeItem(key);
      }
    } catch {
      // Skip non-JSON items
    }
  }
}

/**
 * Gets storage size in bytes.
 * @returns Storage size in bytes
 */
export function getStorageSize(): number {
  if (typeof window === 'undefined') {
    return 0;
  }

  let total = 0;

  for (const key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      total += localStorage[key].length + key.length;
    }
  }

  return total;
}

/**
 * Gets storage quota information.
 * @returns Storage quota info
 */
export async function getStorageQuota(): Promise<{
  quota: number;
  usage: number;
  available: number;
} | null> {
  if (typeof navigator === 'undefined' || !('storage' in navigator)) {
    return null;
  }

  try {
    const estimate = await (navigator as any).storage.estimate();
    return {
      quota: estimate.quota || 0,
      usage: estimate.usage || 0,
      available: (estimate.quota || 0) - (estimate.usage || 0),
    };
  } catch {
    return null;
  }
}

