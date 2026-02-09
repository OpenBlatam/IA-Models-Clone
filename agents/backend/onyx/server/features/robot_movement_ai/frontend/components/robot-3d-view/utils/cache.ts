/**
 * Intelligent caching system
 * @module robot-3d-view/utils/cache
 */

/**
 * Cache entry
 */
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
  accessCount: number;
  lastAccessed: number;
}

/**
 * Cache options
 */
export interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  maxSize?: number; // Maximum cache size
  strategy?: 'lru' | 'lfu' | 'fifo'; // Eviction strategy
}

/**
 * Cache Manager class
 */
export class CacheManager<T> {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private options: Required<CacheOptions>;

  constructor(options: CacheOptions = {}) {
    this.options = {
      ttl: options.ttl ?? 3600000, // 1 hour default
      maxSize: options.maxSize ?? 100,
      strategy: options.strategy ?? 'lru',
    };
  }

  /**
   * Sets a value in cache
   */
  set(key: string, value: T, ttl?: number): void {
    const now = Date.now();
    const entryTTL = ttl ?? this.options.ttl;
    const expiresAt = now + entryTTL;

    // Evict if cache is full
    if (this.cache.size >= this.options.maxSize) {
      this.evict();
    }

    this.cache.set(key, {
      data: value,
      timestamp: now,
      expiresAt,
      accessCount: 0,
      lastAccessed: now,
    });
  }

  /**
   * Gets a value from cache
   */
  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    // Update access info
    entry.accessCount++;
    entry.lastAccessed = Date.now();

    return entry.data;
  }

  /**
   * Checks if key exists in cache
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Deletes a key from cache
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clears all cache
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Gets cache size
   */
  size(): number {
    this.cleanExpired();
    return this.cache.size;
  }

  /**
   * Evicts an entry based on strategy
   */
  private evict(): void {
    this.cleanExpired();

    if (this.cache.size < this.options.maxSize) return;

    let entryToEvict: string | null = null;
    let evictValue: number | null = null;

    switch (this.options.strategy) {
      case 'lru': {
        // Least Recently Used
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.lastAccessed < evictValue) {
            evictValue = entry.lastAccessed;
            entryToEvict = key;
          }
        }
        break;
      }

      case 'lfu': {
        // Least Frequently Used
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.accessCount < evictValue) {
            evictValue = entry.accessCount;
            entryToEvict = key;
          }
        }
        break;
      }

      case 'fifo': {
        // First In First Out
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.timestamp < evictValue) {
            evictValue = entry.timestamp;
            entryToEvict = key;
          }
        }
        break;
      }
    }

    if (entryToEvict) {
      this.cache.delete(entryToEvict);
    }
  }

  /**
   * Cleans expired entries
   */
  private cleanExpired(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Gets cache statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    strategy: string;
    hitRate?: number;
  } {
    this.cleanExpired();
    return {
      size: this.cache.size,
      maxSize: this.options.maxSize,
      strategy: this.options.strategy,
    };
  }
}

/**
 * Global cache instances
 */
export const configCache = new CacheManager<unknown>({
  ttl: 3600000, // 1 hour
  maxSize: 50,
  strategy: 'lru',
});

export const trajectoryCache = new CacheManager<unknown>({
  ttl: 1800000, // 30 minutes
  maxSize: 100,
  strategy: 'lru',
});



