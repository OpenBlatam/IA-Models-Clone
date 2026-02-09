/**
 * Simple in-memory cache utilities
 * 
 * Provides caching mechanisms for API responses and computed values
 */

/**
 * Cache entry with expiration
 */
interface CacheEntry<T> {
  readonly data: T;
  readonly expiresAt: number;
}

/**
 * In-memory cache implementation
 */
class MemoryCache {
  private readonly cache = new Map<string, CacheEntry<unknown>>();

  /**
   * Gets value from cache if not expired
   */
  get<T>(key: string): T | null {
    const entry = this.cache.get(key) as CacheEntry<T> | undefined;

    if (!entry) {
      return null;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  /**
   * Sets value in cache with TTL
   */
  set<T>(key: string, value: T, ttlMs: number): void {
    const expiresAt = Date.now() + ttlMs;
    this.cache.set(key, { data: value, expiresAt });
  }

  /**
   * Deletes value from cache
   */
  delete(key: string): void {
    this.cache.delete(key);
  }

  /**
   * Clears all cache entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Clears expired entries
   */
  clearExpired(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Gets cache size
   */
  size(): number {
    return this.cache.size;
  }
}

// Singleton cache instance
const cache = new MemoryCache();

/**
 * Gets value from cache
 */
export function getCache<T>(key: string): T | null {
  return cache.get<T>(key);
}

/**
 * Sets value in cache with TTL
 */
export function setCache<T>(key: string, value: T, ttlMs: number): void {
  cache.set(key, value, ttlMs);
}

/**
 * Deletes value from cache
 */
export function deleteCache(key: string): void {
  cache.delete(key);
}

/**
 * Clears all cache
 */
export function clearCache(): void {
  cache.clear();
}

/**
 * Clears expired cache entries
 */
export function clearExpiredCache(): void {
  cache.clearExpired();
}

/**
 * Cache key builder
 */
export function buildCacheKey(...parts: (string | number)[]): string {
  return `agent:${parts.join(":")}`;
}

/**
 * Creates a cached function that caches results
 */
export function cached<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  keyBuilder: (...args: TArgs) => string,
  ttlMs: number
): (...args: TArgs) => Promise<TReturn> {
  return async (...args: TArgs): Promise<TReturn> => {
    const key = keyBuilder(...args);
    const cached = getCache<TReturn>(key);

    if (cached !== null) {
      return cached;
    }

    const result = await fn(...args);
    setCache(key, result, ttlMs);
    return result;
  };
}




