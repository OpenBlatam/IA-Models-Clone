import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * Storage utilities with error handling
 */

/**
 * Get item from storage
 */
export const getStorageItem = async <T,>(
  key: string,
  defaultValue?: T
): Promise<T | null> => {
  try {
    const item = await AsyncStorage.getItem(key);
    if (item === null) return defaultValue || null;
    return JSON.parse(item) as T;
  } catch (error) {
    console.error(`Error getting storage item ${key}:`, error);
    return defaultValue || null;
  }
};

/**
 * Set item in storage
 */
export const setStorageItem = async <T,>(
  key: string,
  value: T
): Promise<boolean> => {
  try {
    await AsyncStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Error setting storage item ${key}:`, error);
    return false;
  }
};

/**
 * Remove item from storage
 */
export const removeStorageItem = async (key: string): Promise<boolean> => {
  try {
    await AsyncStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error(`Error removing storage item ${key}:`, error);
    return false;
  }
};

/**
 * Clear all storage
 */
export const clearStorage = async (): Promise<boolean> => {
  try {
    await AsyncStorage.clear();
    return true;
  } catch (error) {
    console.error('Error clearing storage:', error);
    return false;
  }
};

/**
 * Get all keys from storage
 */
export const getAllStorageKeys = async (): Promise<string[]> => {
  try {
    return await AsyncStorage.getAllKeys();
  } catch (error) {
    console.error('Error getting storage keys:', error);
    return [];
  }
};

/**
 * Get multiple items from storage
 */
export const getMultipleStorageItems = async <T,>(
  keys: string[]
): Promise<Record<string, T | null>> => {
  try {
    const items = await AsyncStorage.multiGet(keys);
    const result: Record<string, T | null> = {};
    items.forEach(([key, value]) => {
      result[key] = value ? (JSON.parse(value) as T) : null;
    });
    return result;
  } catch (error) {
    console.error('Error getting multiple storage items:', error);
    return {};
  }
};

/**
 * Set multiple items in storage
 */
export const setMultipleStorageItems = async <T,>(
  items: Record<string, T>
): Promise<boolean> => {
  try {
    const entries = Object.entries(items).map(([key, value]) => [
      key,
      JSON.stringify(value),
    ]);
    await AsyncStorage.multiSet(entries);
    return true;
  } catch (error) {
    console.error('Error setting multiple storage items:', error);
    return false;
  }
};

