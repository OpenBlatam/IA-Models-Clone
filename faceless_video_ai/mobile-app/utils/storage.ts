import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';

export const StorageKeys = {
  AUTH_TOKEN: 'auth_token',
  API_KEY: 'api_key',
  THEME: 'theme',
  LANGUAGE: 'language',
  USER_PREFERENCES: 'user_preferences',
  CACHE_TIMESTAMP: 'cache_timestamp',
} as const;

export class SecureStorage {
  static async setItem(key: string, value: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(key, value);
    } catch (error) {
      console.error(`Failed to set secure item ${key}:`, error);
      throw error;
    }
  }

  static async getItem(key: string): Promise<string | null> {
    try {
      return await SecureStore.getItemAsync(key);
    } catch (error) {
      console.error(`Failed to get secure item ${key}:`, error);
      return null;
    }
  }

  static async removeItem(key: string): Promise<void> {
    try {
      await SecureStore.deleteItemAsync(key);
    } catch (error) {
      console.error(`Failed to remove secure item ${key}:`, error);
    }
  }

  static async clear(): Promise<void> {
    try {
      // SecureStore doesn't have a clear method, so we need to remove items individually
      // In a real app, you'd track which keys are stored
      await SecureStore.deleteItemAsync(StorageKeys.AUTH_TOKEN);
      await SecureStore.deleteItemAsync(StorageKeys.API_KEY);
    } catch (error) {
      console.error('Failed to clear secure storage:', error);
    }
  }
}

export class AppStorage {
  static async setItem<T>(key: string, value: T): Promise<void> {
    try {
      const jsonValue = JSON.stringify(value);
      await AsyncStorage.setItem(key, jsonValue);
    } catch (error) {
      console.error(`Failed to set item ${key}:`, error);
      throw error;
    }
  }

  static async getItem<T>(key: string): Promise<T | null> {
    try {
      const jsonValue = await AsyncStorage.getItem(key);
      return jsonValue != null ? (JSON.parse(jsonValue) as T) : null;
    } catch (error) {
      console.error(`Failed to get item ${key}:`, error);
      return null;
    }
  }

  static async removeItem(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(key);
    } catch (error) {
      console.error(`Failed to remove item ${key}:`, error);
    }
  }

  static async clear(): Promise<void> {
    try {
      await AsyncStorage.clear();
    } catch (error) {
      console.error('Failed to clear storage:', error);
    }
  }

  static async getAllKeys(): Promise<string[]> {
    try {
      return await AsyncStorage.getAllKeys();
    } catch (error) {
      console.error('Failed to get all keys:', error);
      return [];
    }
  }

  static async multiGet(keys: string[]): Promise<Array<[string, string | null]>> {
    try {
      return await AsyncStorage.multiGet(keys);
    } catch (error) {
      console.error('Failed to multi get:', error);
      return [];
    }
  }

  static async multiSet(items: Array<[string, string]>): Promise<void> {
    try {
      await AsyncStorage.multiSet(items);
    } catch (error) {
      console.error('Failed to multi set:', error);
      throw error;
    }
  }
}


