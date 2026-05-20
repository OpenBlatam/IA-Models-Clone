// Offline cache utilities

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiry: number;
}

class OfflineCache {
  private cache: Map<string, CacheEntry<any>> = new Map();
  private defaultExpiry = 5 * 60 * 1000; // 5 minutes

  set<T>(key: string, data: T, expiry?: number): void {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      expiry: expiry || this.defaultExpiry,
    };
    this.cache.set(key, entry);
    this.saveToStorage();
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) {
      return null;
    }

    const now = Date.now();
    if (now - entry.timestamp > entry.expiry) {
      this.cache.delete(key);
      this.saveToStorage();
      return null;
    }

    return entry.data as T;
  }

  has(key: string): boolean {
    return this.cache.has(key) && this.get(key) !== null;
  }

  delete(key: string): void {
    this.cache.delete(key);
    this.saveToStorage();
  }

  clear(): void {
    this.cache.clear();
    this.saveToStorage();
  }

  private saveToStorage(): void {
    if (typeof window !== 'undefined') {
      try {
        const data = Array.from(this.cache.entries()).map(([key, entry]) => [
          key,
          entry,
        ]);
        localStorage.setItem('offline-cache', JSON.stringify(data));
      } catch (error) {
        console.error('Failed to save cache to storage:', error);
      }
    }
  }

  loadFromStorage(): void {
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem('offline-cache');
        if (stored) {
          const data = JSON.parse(stored);
          this.cache = new Map(data);
        }
      } catch (error) {
        console.error('Failed to load cache from storage:', error);
      }
    }
  }
}

export const offlineCache = new OfflineCache();

// Initialize cache on load
if (typeof window !== 'undefined') {
  offlineCache.loadFromStorage();
}

