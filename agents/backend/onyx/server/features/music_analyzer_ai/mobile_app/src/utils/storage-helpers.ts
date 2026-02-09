import AsyncStorage from '@react-native-async-storage/async-storage';
import EncryptedStorage from 'react-native-encrypted-storage';

/**
 * Storage helpers with automatic encryption for sensitive data
 */

/**
 * Store data securely (encrypted) or in regular storage
 */
export async function storeData(
  key: string,
  value: unknown,
  encrypted = false
): Promise<void> {
  try {
    const stringValue = JSON.stringify(value);

    if (encrypted) {
      await EncryptedStorage.setItem(key, stringValue);
    } else {
      await AsyncStorage.setItem(key, stringValue);
    }
  } catch (error) {
    console.error(`Error storing ${key}:`, error);
    throw error;
  }
}

/**
 * Retrieve data from secure or regular storage
 */
export async function getData<T>(
  key: string,
  encrypted = false
): Promise<T | null> {
  try {
    const value = encrypted
      ? await EncryptedStorage.getItem(key)
      : await AsyncStorage.getItem(key);

    if (value === null) {
      return null;
    }

    return JSON.parse(value) as T;
  } catch (error) {
    console.error(`Error retrieving ${key}:`, error);
    return null;
  }
}

/**
 * Remove data from storage
 */
export async function removeData(key: string, encrypted = false): Promise<void> {
  try {
    if (encrypted) {
      await EncryptedStorage.removeItem(key);
    } else {
      await AsyncStorage.removeItem(key);
    }
  } catch (error) {
    console.error(`Error removing ${key}:`, error);
    throw error;
  }
}

/**
 * Clear all data from storage
 */
export async function clearStorage(encrypted = false): Promise<void> {
  try {
    if (encrypted) {
      await EncryptedStorage.clear();
    } else {
      await AsyncStorage.clear();
    }
  } catch (error) {
    console.error('Error clearing storage:', error);
    throw error;
  }
}

/**
 * Get all keys from storage
 */
export async function getAllKeys(encrypted = false): Promise<string[]> {
  try {
    if (encrypted) {
      // EncryptedStorage doesn't have getAllKeys, so we can't implement this
      return [];
    }
    return await AsyncStorage.getAllKeys();
  } catch (error) {
    console.error('Error getting all keys:', error);
    return [];
  }
}

/**
 * Check if key exists in storage
 */
export async function hasKey(key: string, encrypted = false): Promise<boolean> {
  try {
    const value = encrypted
      ? await EncryptedStorage.getItem(key)
      : await AsyncStorage.getItem(key);
    return value !== null;
  } catch {
    return false;
  }
}

