import AsyncStorage from '@react-native-async-storage/async-storage';

interface CacheItem<T> {
  data: T;
  timestamp: number;
  expiry: number;
}

/**
 * Cache manager with TTL (Time To Live) support
 * Automatically expires old entries
 */
class CacheManager {
  private prefix = '@cache:';

  private getKey(key: string): string {
    return `${this.prefix}${key}`;
  }

  async set<T>(key: string, data: T, ttlMs: number): Promise<void> {
    try {
      const item: CacheItem<T> = {
        data,
        timestamp: Date.now(),
        expiry: Date.now() + ttlMs,
      };
      await AsyncStorage.setItem(this.getKey(key), JSON.stringify(item));
    } catch (error) {
      console.error(`Cache set error for ${key}:`, error);
    }
  }

  async get<T>(key: string): Promise<T | null> {
    try {
      const itemStr = await AsyncStorage.getItem(this.getKey(key));
      if (!itemStr) {
        return null;
      }

      const item = JSON.parse(itemStr) as CacheItem<T>;

      if (Date.now() > item.expiry) {
        await this.remove(key);
        return null;
      }

      return item.data;
    } catch (error) {
      console.error(`Cache get error for ${key}:`, error);
      return null;
    }
  }

  async remove(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.getKey(key));
    } catch (error) {
      console.error(`Cache remove error for ${key}:`, error);
    }
  }

  async clear(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter((key) => key.startsWith(this.prefix));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Cache clear error:', error);
    }
  }

  async getAllKeys(): Promise<string[]> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      return keys
        .filter((key) => key.startsWith(this.prefix))
        .map((key) => key.replace(this.prefix, ''));
    } catch (error) {
      console.error('Cache getAllKeys error:', error);
      return [];
    }
  }
}

export const cacheManager = new CacheManager();

