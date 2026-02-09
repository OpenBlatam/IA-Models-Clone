/**
 * Advanced storage helper functions
 * Enhanced storage operations with encryption and compression
 */

import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * Secure storage operations
 */
export const secureStorage = {
  async setItem(key: string, value: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(key, value);
    } catch (error) {
      console.error('Secure storage set error:', error);
      throw error;
    }
  },

  async getItem(key: string): Promise<string | null> {
    try {
      return await SecureStore.getItemAsync(key);
    } catch (error) {
      console.error('Secure storage get error:', error);
      return null;
    }
  },

  async removeItem(key: string): Promise<void> {
    try {
      await SecureStore.deleteItemAsync(key);
    } catch (error) {
      console.error('Secure storage remove error:', error);
      throw error;
    }
  },

  async clear(): Promise<void> {
    // Note: SecureStore doesn't have a clear method
    // You would need to track keys and remove them individually
    console.warn('SecureStore.clear() is not available');
  },
};

/**
 * Storage with JSON serialization
 */
export const jsonStorage = {
  async setItem<T>(key: string, value: T): Promise<void> {
    try {
      const json = JSON.stringify(value);
      await AsyncStorage.setItem(key, json);
    } catch (error) {
      console.error('JSON storage set error:', error);
      throw error;
    }
  },

  async getItem<T>(key: string): Promise<T | null> {
    try {
      const json = await AsyncStorage.getItem(key);
      if (json === null) return null;
      return JSON.parse(json) as T;
    } catch (error) {
      console.error('JSON storage get error:', error);
      return null;
    }
  },

  async removeItem(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(key);
    } catch (error) {
      console.error('JSON storage remove error:', error);
      throw error;
    }
  },
};

/**
 * Storage with expiration
 */
export const expiringStorage = {
  async setItem<T>(
    key: string,
    value: T,
    ttl: number // Time to live in milliseconds
  ): Promise<void> {
    try {
      const item = {
        value,
        expiresAt: Date.now() + ttl,
      };
      await jsonStorage.setItem(key, item);
    } catch (error) {
      console.error('Expiring storage set error:', error);
      throw error;
    }
  },

  async getItem<T>(key: string): Promise<T | null> {
    try {
      const item = await jsonStorage.getItem<{ value: T; expiresAt: number }>(
        key
      );
      if (!item) return null;

      if (Date.now() > item.expiresAt) {
        await jsonStorage.removeItem(key);
        return null;
      }

      return item.value;
    } catch (error) {
      console.error('Expiring storage get error:', error);
      return null;
    }
  },

  async removeItem(key: string): Promise<void> {
    await jsonStorage.removeItem(key);
  },
};

