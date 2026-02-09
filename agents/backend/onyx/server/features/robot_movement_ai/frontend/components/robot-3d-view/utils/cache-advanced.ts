/**
 * Advanced caching system with multiple strategies
 * @module robot-3d-view/utils/cache-advanced
 */

import { CacheManager } from './cache';

/**
 * Cache strategy
 */
export type CacheStrategy = 'lru' | 'lfu' | 'fifo' | 'ttl' | 'adaptive';

/**
 * Advanced cache entry
 */
interface AdvancedCacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
  accessCount: number;
  lastAccessed: number;
  size: number;
  priority: number;
}

/**
 * Advanced Cache Manager class
 */
export class AdvancedCacheManager<T> {
  private cache: Map<string, AdvancedCacheEntry<T>> = new Map();
  private options: {
    maxSize: number;
    maxMemory: number;
    strategy: CacheStrategy;
    defaultTTL: number;
  };
  private currentMemory = 0;

  constructor(options: {
    maxSize?: number;
    maxMemory?: number;
    strategy?: CacheStrategy;
    defaultTTL?: number;
  } = {}) {
    this.options = {
      maxSize: options.maxSize ?? 1000,
      maxMemory: options.maxMemory ?? 50 * 1024 * 1024, // 50MB
      strategy: options.strategy ?? 'adaptive',
      defaultTTL: options.defaultTTL ?? 3600000, // 1 hour
    };
  }

  /**
   * Sets a value in cache
   */
  set(
    key: string,
    value: T,
    ttl?: number,
    priority = 0
  ): void {
    const now = Date.now();
    const entryTTL = ttl ?? this.options.defaultTTL;
    const expiresAt = now + entryTTL;
    const size = this.estimateSize(value);

    // Check memory limit
    if (this.currentMemory + size > this.options.maxMemory) {
      this.evictByMemory(size);
    }

    // Check size limit
    if (this.cache.size >= this.options.maxSize) {
      this.evict();
    }

    this.cache.set(key, {
      data: value,
      timestamp: now,
      expiresAt,
      accessCount: 0,
      lastAccessed: now,
      size,
      priority,
    });

    this.currentMemory += size;
  }

  /**
   * Gets a value from cache
   */
  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.delete(key);
      return undefined;
    }

    // Update access info
    entry.accessCount++;
    entry.lastAccessed = Date.now();

    return entry.data;
  }

  /**
   * Deletes a key from cache
   */
  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (entry) {
      this.currentMemory -= entry.size;
      return this.cache.delete(key);
    }
    return false;
  }

  /**
   * Clears all cache
   */
  clear(): void {
    this.cache.clear();
    this.currentMemory = 0;
  }

  /**
   * Evicts entries based on strategy
   */
  private evict(): void {
    this.cleanExpired();

    if (this.cache.size < this.options.maxSize) return;

    let entryToEvict: string | null = null;
    let evictValue: number | null = null;

    switch (this.options.strategy) {
      case 'lru':
        // Least Recently Used
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.lastAccessed < evictValue) {
            evictValue = entry.lastAccessed;
            entryToEvict = key;
          }
        }
        break;

      case 'lfu':
        // Least Frequently Used
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.accessCount < evictValue) {
            evictValue = entry.accessCount;
            entryToEvict = key;
          }
        }
        break;

      case 'fifo':
        // First In First Out
        for (const [key, entry] of this.cache.entries()) {
          if (evictValue === null || entry.timestamp < evictValue) {
            evictValue = entry.timestamp;
            entryToEvict = key;
          }
        }
        break;

      case 'adaptive':
        // Adaptive: combination of LRU and priority
        for (const [key, entry] of this.cache.entries()) {
          const score = entry.lastAccessed - entry.priority * 1000;
          if (evictValue === null || score < evictValue) {
            evictValue = score;
            entryToEvict = key;
          }
        }
        break;
    }

    if (entryToEvict) {
      this.delete(entryToEvict);
    }
  }

  /**
   * Evicts entries by memory
   */
  private evictByMemory(requiredSize: number): void {
    const entries = Array.from(this.cache.entries())
      .sort((a, b) => {
        // Sort by priority and last access
        const scoreA = a[1].lastAccessed - a[1].priority * 1000;
        const scoreB = b[1].lastAccessed - b[1].priority * 1000;
        return scoreA - scoreB;
      });

    for (const [key, entry] of entries) {
      if (this.currentMemory + requiredSize <= this.options.maxMemory) {
        break;
      }
      this.delete(key);
    }
  }

  /**
   * Cleans expired entries
   */
  private cleanExpired(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.delete(key);
      }
    }
  }

  /**
   * Estimates size of value
   */
  private estimateSize(value: T): number {
    // Rough estimation
    const str = JSON.stringify(value);
    return new Blob([str]).size;
  }

  /**
   * Gets cache statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    memory: number;
    maxMemory: number;
    strategy: string;
    hitRate?: number;
  } {
    this.cleanExpired();
    return {
      size: this.cache.size,
      maxSize: this.options.maxSize,
      memory: this.currentMemory,
      maxMemory: this.options.maxMemory,
      strategy: this.options.strategy,
    };
  }
}

/**
 * Global advanced cache instances
 */
export const advancedConfigCache = new AdvancedCacheManager<unknown>({
  maxSize: 100,
  maxMemory: 10 * 1024 * 1024, // 10MB
  strategy: 'adaptive',
  defaultTTL: 3600000, // 1 hour
});

export const advancedTrajectoryCache = new AdvancedCacheManager<unknown>({
  maxSize: 500,
  maxMemory: 50 * 1024 * 1024, // 50MB
  strategy: 'lru',
  defaultTTL: 1800000, // 30 minutes
});



