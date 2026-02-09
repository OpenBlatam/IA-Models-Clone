/**
 * Storage utilities for localStorage and sessionStorage
 */

type StorageType = "localStorage" | "sessionStorage";

const getStorage = (type: StorageType): Storage | null => {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return type === "localStorage" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
};

export const getStorageItem = <T = string>(
  key: string,
  type: StorageType = "localStorage"
): T | null => {
  const storage = getStorage(type);
  if (!storage) {
    return null;
  }

  try {
    const item = storage.getItem(key);
    if (item === null) {
      return null;
    }
    return JSON.parse(item) as T;
  } catch {
    return storage.getItem(key) as T | null;
  }
};

export const setStorageItem = <T>(
  key: string,
  value: T,
  type: StorageType = "localStorage"
): boolean => {
  const storage = getStorage(type);
  if (!storage) {
    return false;
  }

  try {
    const serialized =
      typeof value === "string" ? value : JSON.stringify(value);
    storage.setItem(key, serialized);
    return true;
  } catch {
    return false;
  }
};

export const removeStorageItem = (
  key: string,
  type: StorageType = "localStorage"
): boolean => {
  const storage = getStorage(type);
  if (!storage) {
    return false;
  }

  try {
    storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
};

export const clearStorage = (type: StorageType = "localStorage"): boolean => {
  const storage = getStorage(type);
  if (!storage) {
    return false;
  }

  try {
    storage.clear();
    return true;
  } catch {
    return false;
  }
};

export const getStorageKeys = (
  type: StorageType = "localStorage"
): readonly string[] => {
  const storage = getStorage(type);
  if (!storage) {
    return [];
  }

  try {
    return Array.from({ length: storage.length }, (_, i) => storage.key(i)!).filter(
      (key): key is string => key !== null
    );
  } catch {
    return [];
  }
};

export const hasStorageItem = (
  key: string,
  type: StorageType = "localStorage"
): boolean => {
  const storage = getStorage(type);
  if (!storage) {
    return false;
  }

  try {
    return storage.getItem(key) !== null;
  } catch {
    return false;
  }
};

export const getStorageSize = (
  type: StorageType = "localStorage"
): number => {
  const storage = getStorage(type);
  if (!storage) {
    return 0;
  }

  try {
    let total = 0;
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key) {
        const value = storage.getItem(key) || "";
        total += key.length + value.length;
      }
    }
    return total;
  } catch {
    return 0;
  }
};





