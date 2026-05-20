/**
 * Storage Utilities
 * =================
 * Utilities for async storage operations
 */

import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Secure storage for sensitive data
export async function setSecureItem(key: string, value: string): Promise<void> {
  await SecureStore.setItemAsync(key, value);
}

export async function getSecureItem(key: string): Promise<string | null> {
  return await SecureStore.getItemAsync(key);
}

export async function removeSecureItem(key: string): Promise<void> {
  await SecureStore.deleteItemAsync(key);
}

// Regular storage for non-sensitive data
export async function setItem(key: string, value: string): Promise<void> {
  await AsyncStorage.setItem(key, value);
}

export async function getItem(key: string): Promise<string | null> {
  return await AsyncStorage.getItem(key);
}

export async function removeItem(key: string): Promise<void> {
  await AsyncStorage.removeItem(key);
}

export async function clearStorage(): Promise<void> {
  await AsyncStorage.clear();
}

// JSON helpers
export async function setJSONItem<T>(key: string, value: T): Promise<void> {
  await setItem(key, JSON.stringify(value));
}

export async function getJSONItem<T>(key: string): Promise<T | null> {
  const item = await getItem(key);
  if (!item) return null;
  try {
    return JSON.parse(item) as T;
  } catch {
    return null;
  }
}



