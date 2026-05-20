interface CacheEntry<T> {
  value: T;
  timestamp: number;
  ttl?: number;
}

class MemoryCache {
  private cache = new Map<string, CacheEntry<unknown>>();

  set<T>(key: string, value: T, ttl?: number): void {
    this.cache.set(key, {
      value,
      timestamp: Date.now(),
      ttl,
    });
  }

  get<T>(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) {
      return undefined;
    }

    if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value as T;
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) {
      return false;
    }

    if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

export const memoryCache = new MemoryCache();

export const createCache = <T,>(key: string, fetcher: () => Promise<T>, ttl?: number): (() => Promise<T>) => {
  return async () => {
    const cached = memoryCache.get<T>(key);
    if (cached !== undefined) {
      return cached;
    }

    const value = await fetcher();
    memoryCache.set(key, value, ttl);
    return value;
  };
};



