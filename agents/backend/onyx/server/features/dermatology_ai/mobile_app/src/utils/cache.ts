import AsyncStorage from '@react-native-async-storage/async-storage';

const CACHE_PREFIX = '@dermatology_cache_';
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours

interface CacheItem<T> {
  data: T;
  timestamp: number;
}

/**
 * Cache utility with expiration
 */
export class Cache {
  static async set<T>(key: string, value: T): Promise<void> {
    try {
      const item: CacheItem<T> = {
        data: value,
        timestamp: Date.now(),
      };
      await AsyncStorage.setItem(CACHE_PREFIX + key, JSON.stringify(item));
    } catch (error) {
      console.error('Error setting cache:', error);
    }
  }

  static async get<T>(key: string): Promise<T | null> {
    try {
      const item = await AsyncStorage.getItem(CACHE_PREFIX + key);
      if (!item) return null;

      const parsed: CacheItem<T> = JSON.parse(item);
      const now = Date.now();

      // Check if expired
      if (now - parsed.timestamp > CACHE_EXPIRY) {
        await this.remove(key);
        return null;
      }

      return parsed.data;
    } catch (error) {
      console.error('Error getting cache:', error);
      return null;
    }
  }

  static async remove(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(CACHE_PREFIX + key);
    } catch (error) {
      console.error('Error removing cache:', error);
    }
  }

  static async clear(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith(CACHE_PREFIX));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  }

  static async getAllKeys(): Promise<string[]> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      return keys
        .filter(key => key.startsWith(CACHE_PREFIX))
        .map(key => key.replace(CACHE_PREFIX, ''));
    } catch (error) {
      console.error('Error getting cache keys:', error);
      return [];
    }
  }
}

