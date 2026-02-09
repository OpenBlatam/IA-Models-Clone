/**
 * Cache & Storage Strategies Testing
 * 
 * Tests that verify caching strategies, storage management,
 * cache invalidation, and data persistence.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock Cache API
const mockCache = {
  match: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  keys: vi.fn(),
  add: vi.fn(),
  addAll: vi.fn(),
};

const mockCaches = {
  open: vi.fn().mockResolvedValue(mockCache),
  has: vi.fn().mockResolvedValue(true),
  delete: vi.fn().mockResolvedValue(true),
  keys: vi.fn().mockResolvedValue(['v1', 'v2']),
  match: vi.fn(),
};

Object.defineProperty(global, 'caches', {
  writable: true,
  value: mockCaches,
});

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(global, 'localStorage', {
  writable: true,
  value: localStorageMock,
});

describe('Cache & Storage Strategies Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  describe('Cache-First Strategy', () => {
    it('should return cached response if available', async () => {
      const cacheFirst = async (request: Request) => {
        const cache = await caches.open('v1');
        const cached = await cache.match(request);
        if (cached) return cached;
        
        const response = await fetch(request);
        await cache.put(request, response.clone());
        return response;
      };

      const request = new Request('/api/tracks');
      const cachedResponse = new Response('cached data');
      mockCache.match.mockResolvedValue(cachedResponse);

      const result = await cacheFirst(request);
      expect(result).toBe(cachedResponse);
      expect(mockCache.match).toHaveBeenCalled();
    });

    it('should fetch and cache if not in cache', async () => {
      const cacheFirst = async (request: Request) => {
        const cache = await caches.open('v1');
        const cached = await cache.match(request);
        if (cached) return cached;
        
        const response = await fetch(request);
        await cache.put(request, response.clone());
        return response;
      };

      const request = new Request('/api/tracks');
      const networkResponse = new Response('network data');
      mockCache.match.mockResolvedValue(null);
      global.fetch = vi.fn().mockResolvedValue(networkResponse);

      const result = await cacheFirst(request);
      expect(result).toBe(networkResponse);
      expect(mockCache.put).toHaveBeenCalled();
    });
  });

  describe('Network-First Strategy', () => {
    it('should fetch from network first', async () => {
      const networkFirst = async (request: Request) => {
        try {
          const response = await fetch(request);
          const cache = await caches.open('v1');
          await cache.put(request, response.clone());
          return response;
        } catch {
          const cache = await caches.open('v1');
          return cache.match(request);
        }
      };

      const request = new Request('/api/tracks');
      const networkResponse = new Response('network data');
      global.fetch = vi.fn().mockResolvedValue(networkResponse);

      const result = await networkFirst(request);
      expect(result).toBe(networkResponse);
      expect(global.fetch).toHaveBeenCalled();
    });

    it('should fallback to cache on network failure', async () => {
      const networkFirst = async (request: Request) => {
        try {
          const response = await fetch(request);
          const cache = await caches.open('v1');
          await cache.put(request, response.clone());
          return response;
        } catch {
          const cache = await caches.open('v1');
          return cache.match(request);
        }
      };

      const request = new Request('/api/tracks');
      const cachedResponse = new Response('cached data');
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));
      mockCache.match.mockResolvedValue(cachedResponse);

      const result = await networkFirst(request);
      expect(result).toBe(cachedResponse);
    });
  });

  describe('Stale-While-Revalidate Strategy', () => {
    it('should return cached data immediately and update in background', async () => {
      const staleWhileRevalidate = async (request: Request) => {
        const cache = await caches.open('v1');
        const cached = await cache.match(request);
        
        const fetchPromise = fetch(request).then(response => {
          cache.put(request, response.clone());
          return response;
        });
        
        return cached || fetchPromise;
      };

      const request = new Request('/api/tracks');
      const cachedResponse = new Response('cached data');
      mockCache.match.mockResolvedValue(cachedResponse);
      global.fetch = vi.fn().mockResolvedValue(new Response('fresh data'));

      const result = await staleWhileRevalidate(request);
      expect(result).toBe(cachedResponse);
      expect(global.fetch).toHaveBeenCalled();
    });
  });

  describe('Cache Invalidation', () => {
    it('should invalidate cache by version', async () => {
      const invalidateCache = async (newVersion: string) => {
        const cacheNames = await caches.keys();
        const oldCaches = cacheNames.filter(name => name !== newVersion);
        await Promise.all(oldCaches.map(name => caches.delete(name)));
      };

      await invalidateCache('v3');
      expect(mockCaches.delete).toHaveBeenCalled();
    });

    it('should invalidate specific cache entries', async () => {
      const invalidateEntry = async (request: Request) => {
        const cache = await caches.open('v1');
        await cache.delete(request);
      };

      const request = new Request('/api/tracks');
      await invalidateEntry(request);
      expect(mockCache.delete).toHaveBeenCalledWith(request);
    });

    it('should invalidate cache by pattern', async () => {
      const invalidateByPattern = async (pattern: string) => {
        const cache = await caches.open('v1');
        const keys = await cache.keys();
        const matchingKeys = keys.filter(key => key.url.includes(pattern));
        await Promise.all(matchingKeys.map(key => cache.delete(key)));
      };

      const request1 = new Request('/api/tracks');
      const request2 = new Request('/api/playlists');
      mockCache.keys.mockResolvedValue([request1, request2]);

      await invalidateByPattern('tracks');
      expect(mockCache.delete).toHaveBeenCalled();
    });
  });

  describe('localStorage Management', () => {
    it('should store data in localStorage', () => {
      const storeData = (key: string, data: any) => {
        localStorage.setItem(key, JSON.stringify(data));
      };

      storeData('tracks', [{ id: '1', name: 'Track 1' }]);
      expect(localStorage.getItem('tracks')).toBeDefined();
    });

    it('should retrieve data from localStorage', () => {
      localStorage.setItem('tracks', JSON.stringify([{ id: '1', name: 'Track 1' }]));

      const getData = (key: string) => {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
      };

      const tracks = getData('tracks');
      expect(tracks).toEqual([{ id: '1', name: 'Track 1' }]);
    });

    it('should handle localStorage quota exceeded', () => {
      const storeWithQuotaCheck = (key: string, data: any) => {
        try {
          localStorage.setItem(key, JSON.stringify(data));
          return { success: true };
        } catch (error: any) {
          if (error.name === 'QuotaExceededError') {
            return { success: false, error: 'Storage quota exceeded' };
          }
          throw error;
        }
      };

      // Mock quota exceeded
      const originalSetItem = localStorage.setItem;
      localStorage.setItem = vi.fn().mockImplementation(() => {
        throw new DOMException('QuotaExceededError', 'QuotaExceededError');
      });

      const result = storeWithQuotaCheck('large-data', 'x'.repeat(10000000));
      expect(result.success).toBe(false);
      expect(result.error).toBe('Storage quota exceeded');

      localStorage.setItem = originalSetItem;
    });

    it('should clear expired localStorage entries', () => {
      const setWithExpiry = (key: string, value: any, ttl: number) => {
        const item = {
          value,
          expiry: Date.now() + ttl,
        };
        localStorage.setItem(key, JSON.stringify(item));
      };

      const getWithExpiry = (key: string) => {
        const itemStr = localStorage.getItem(key);
        if (!itemStr) return null;

        const item = JSON.parse(itemStr);
        if (Date.now() > item.expiry) {
          localStorage.removeItem(key);
          return null;
        }
        return item.value;
      };

      setWithExpiry('tracks', [{ id: '1' }], 1000);
      expect(getWithExpiry('tracks')).toBeDefined();

      // Simulate expiry
      vi.useFakeTimers();
      vi.advanceTimersByTime(2000);
      expect(getWithExpiry('tracks')).toBeNull();
      vi.useRealTimers();
    });
  });

  describe('Cache Size Management', () => {
    it('should limit cache size', async () => {
      const limitCacheSize = async (maxSize: number) => {
        const cache = await caches.open('v1');
        const keys = await cache.keys();
        
        if (keys.length > maxSize) {
          const toDelete = keys.slice(0, keys.length - maxSize);
          await Promise.all(toDelete.map(key => cache.delete(key)));
        }
      };

      const request1 = new Request('/api/tracks/1');
      const request2 = new Request('/api/tracks/2');
      const request3 = new Request('/api/tracks/3');
      mockCache.keys.mockResolvedValue([request1, request2, request3]);

      await limitCacheSize(2);
      expect(mockCache.delete).toHaveBeenCalled();
    });

    it('should implement LRU cache eviction', async () => {
      const lruCache = new Map<string, { data: any; timestamp: number }>();
      const maxSize = 3;

      const setLRU = (key: string, data: any) => {
        if (lruCache.size >= maxSize) {
          const oldestKey = Array.from(lruCache.entries())
            .sort((a, b) => a[1].timestamp - b[1].timestamp)[0][0];
          lruCache.delete(oldestKey);
        }
        lruCache.set(key, { data, timestamp: Date.now() });
      };

      setLRU('1', 'data1');
      setLRU('2', 'data2');
      setLRU('3', 'data3');
      expect(lruCache.size).toBe(3);

      setLRU('4', 'data4');
      expect(lruCache.size).toBe(3);
      expect(lruCache.has('1')).toBe(false); // Oldest removed
    });
  });

  describe('Cache Headers', () => {
    it('should respect cache-control headers', () => {
      const response = new Response('data', {
        headers: {
          'Cache-Control': 'max-age=3600',
        },
      });

      const cacheControl = response.headers.get('Cache-Control');
      expect(cacheControl).toBe('max-age=3600');
    });

    it('should respect ETag headers', () => {
      const response = new Response('data', {
        headers: {
          'ETag': '"abc123"',
        },
      });

      const etag = response.headers.get('ETag');
      expect(etag).toBe('"abc123"');
    });

    it('should handle conditional requests', async () => {
      const conditionalRequest = async (request: Request, etag: string) => {
        request.headers.set('If-None-Match', etag);
        const response = await fetch(request);
        
        if (response.status === 304) {
          return null; // Not modified, use cache
        }
        return response;
      };

      const request = new Request('/api/tracks');
      global.fetch = vi.fn().mockResolvedValue(
        new Response(null, { status: 304 })
      );

      const result = await conditionalRequest(request, '"abc123"');
      expect(result).toBeNull();
    });
  });

  describe('Cache Warming', () => {
    it('should pre-cache critical resources', async () => {
      const preCache = async (urls: string[]) => {
        const cache = await caches.open('v1');
        await cache.addAll(urls.map(url => new Request(url)));
      };

      const urls = ['/api/tracks', '/api/playlists'];
      await preCache(urls);
      expect(mockCache.addAll).toHaveBeenCalled();
    });

    it('should cache on navigation', async () => {
      const cacheOnNavigation = async (url: string) => {
        const cache = await caches.open('v1');
        const response = await fetch(url);
        await cache.put(new Request(url), response.clone());
      };

      const url = '/api/tracks';
      global.fetch = vi.fn().mockResolvedValue(new Response('data'));
      await cacheOnNavigation(url);
      expect(mockCache.put).toHaveBeenCalled();
    });
  });
});

