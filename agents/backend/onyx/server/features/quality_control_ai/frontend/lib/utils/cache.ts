interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  maxSize?: number; // Maximum number of entries
}

interface CacheEntry<T> {
  value: T;
  expiresAt?: number;
}

class Cache<K, V> {
  private cache: Map<K, CacheEntry<V>>;
  private ttl?: number;
  private maxSize?: number;

  constructor(options: CacheOptions = {}) {
    this.cache = new Map();
    this.ttl = options.ttl;
    this.maxSize = options.maxSize;
  }

  set(key: K, value: V): void {
    // Remove oldest entries if max size reached
    if (this.maxSize && this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    const entry: CacheEntry<V> = {
      value,
      expiresAt: this.ttl ? Date.now() + this.ttl : undefined,
    };

    this.cache.set(key, entry);
  }

  get(key: K): V | undefined {
    const entry = this.cache.get(key);

    if (!entry) return undefined;

    // Check if expired
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value;
  }

  has(key: K): boolean {
    return this.get(key) !== undefined;
  }

  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }

  keys(): K[] {
    return Array.from(this.cache.keys());
  }

  values(): V[] {
    return Array.from(this.cache.values()).map((entry) => entry.value);
  }

  entries(): [K, V][] {
    return Array.from(this.cache.entries()).map(([key, entry]) => [key, entry.value]);
  }
}

export const createCache = <K, V>(options?: CacheOptions): Cache<K, V> => {
  return new Cache<K, V>(options);
};

