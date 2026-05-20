/**
 * Cache utility functions.
 * Provides helper functions for caching data.
 */

/**
 * Cache entry with expiration.
 */
interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

/**
 * Simple in-memory cache with TTL support.
 */
export class Cache<K, V> {
  private cache = new Map<K, CacheEntry<V>>();
  private defaultTTL: number;

  constructor(defaultTTL: number = 60000) {
    this.defaultTTL = defaultTTL;
  }

  /**
   * Sets value in cache with optional TTL.
   */
  set(key: K, value: V, ttl?: number): void {
    const expiresAt = Date.now() + (ttl ?? this.defaultTTL);
    this.cache.set(key, { value, expiresAt });
  }

  /**
   * Gets value from cache.
   */
  get(key: K): V | undefined {
    const entry = this.cache.get(key);
    if (!entry) {
      return undefined;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value;
  }

  /**
   * Checks if key exists in cache.
   */
  has(key: K): boolean {
    return this.get(key) !== undefined;
  }

  /**
   * Deletes key from cache.
   */
  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clears all cache entries.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Gets cache size.
   */
  size(): number {
    return this.cache.size;
  }

  /**
   * Cleans expired entries.
   */
  clean(): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
        cleaned++;
      }
    }

    return cleaned;
  }
}

/**
 * LRU Cache implementation.
 */
export class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  /**
   * Gets value from cache and moves to end.
   */
  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  /**
   * Sets value in cache.
   */
  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      // Update existing
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove least recently used (first item)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, value);
  }

  /**
   * Checks if key exists in cache.
   */
  has(key: K): boolean {
    return this.cache.has(key);
  }

  /**
   * Deletes key from cache.
   */
  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clears all cache entries.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Gets cache size.
   */
  size(): number {
    return this.cache.size;
  }
}


