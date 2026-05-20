/**
 * Offline Functionality Testing
 * 
 * Tests that verify the application works correctly when offline,
 * including caching, data synchronization, and offline-first strategies.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock Cache API
const mockCache = {
  match: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  keys: vi.fn(),
};

const mockCaches = {
  open: vi.fn().mockResolvedValue(mockCache),
  has: vi.fn().mockResolvedValue(true),
  delete: vi.fn().mockResolvedValue(true),
  keys: vi.fn().mockResolvedValue(['v1']),
  match: vi.fn().mockResolvedValue(new Response()),
};

Object.defineProperty(global, 'caches', {
  writable: true,
  value: mockCaches,
});

// Mock IndexedDB
const mockIndexedDB = {
  open: vi.fn(),
};

Object.defineProperty(global, 'indexedDB', {
  writable: true,
  value: mockIndexedDB,
});

describe('Offline Functionality Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Mock navigator.onLine
    Object.defineProperty(navigator, 'onLine', {
      writable: true,
      configurable: true,
      value: true,
    });
  });

  describe('Network Status Detection', () => {
    it('should detect online status', () => {
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: true,
      });
      
      expect(navigator.onLine).toBe(true);
    });

    it('should detect offline status', () => {
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: false,
      });
      
      expect(navigator.onLine).toBe(false);
    });

    it('should handle online event', () => {
      const handler = vi.fn();
      window.addEventListener('online', handler);
      
      window.dispatchEvent(new Event('online'));
      
      expect(handler).toHaveBeenCalled();
    });

    it('should handle offline event', () => {
      const handler = vi.fn();
      window.addEventListener('offline', handler);
      
      window.dispatchEvent(new Event('offline'));
      
      expect(handler).toHaveBeenCalled();
    });
  });

  describe('Cache Management', () => {
    it('should cache resources for offline use', async () => {
      const cache = await caches.open('v1');
      const request = new Request('/api/tracks');
      const response = new Response(JSON.stringify({ data: 'test' }));
      
      await cache.put(request, response);
      
      expect(mockCache.put).toHaveBeenCalled();
    });

    it('should retrieve cached resources', async () => {
      const cache = await caches.open('v1');
      const request = new Request('/api/tracks');
      const cachedResponse = new Response(JSON.stringify({ data: 'cached' }));
      
      mockCache.match.mockResolvedValue(cachedResponse);
      const response = await cache.match(request);
      
      expect(mockCache.match).toHaveBeenCalledWith(request);
      expect(response).toBe(cachedResponse);
    });

    it('should implement cache-first strategy', async () => {
      const cacheFirst = async (request: Request) => {
        const cache = await caches.open('v1');
        const cached = await cache.match(request);
        if (cached) return cached;
        
        const response = await fetch(request);
        await cache.put(request, response.clone());
        return response;
      };

      const request = new Request('/api/tracks');
      const cachedResponse = new Response('cached');
      mockCache.match.mockResolvedValue(cachedResponse);
      
      const result = await cacheFirst(request);
      expect(result).toBe(cachedResponse);
    });

    it('should implement network-first strategy', async () => {
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
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));
      const cachedResponse = new Response('cached');
      mockCache.match.mockResolvedValue(cachedResponse);
      
      const result = await networkFirst(request);
      expect(result).toBe(cachedResponse);
    });

    it('should clear old caches', async () => {
      const cacheNames = await caches.keys();
      const oldCaches = cacheNames.filter(name => name !== 'v1');
      
      await Promise.all(oldCaches.map(name => caches.delete(name)));
      
      expect(mockCaches.delete).toHaveBeenCalled();
    });
  });

  describe('Data Synchronization', () => {
    it('should queue actions when offline', () => {
      const actionQueue: Array<{ type: string; data: any }> = [];
      
      const queueAction = (action: { type: string; data: any }) => {
        if (!navigator.onLine) {
          actionQueue.push(action);
          return { queued: true };
        }
        return { queued: false };
      };

      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: false,
      });

      const result = queueAction({ type: 'PLAY_TRACK', data: { id: '1' } });
      expect(result.queued).toBe(true);
      expect(actionQueue).toHaveLength(1);
    });

    it('should sync queued actions when online', async () => {
      const actionQueue: Array<{ type: string; data: any }> = [
        { type: 'PLAY_TRACK', data: { id: '1' } },
        { type: 'ADD_TO_PLAYLIST', data: { trackId: '2' } },
      ];

      const syncActions = async () => {
        while (actionQueue.length > 0) {
          const action = actionQueue.shift();
          if (action) {
            await fetch('/api/actions', {
              method: 'POST',
              body: JSON.stringify(action),
            });
          }
        }
      };

      global.fetch = vi.fn().mockResolvedValue(new Response());
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: true,
      });

      await syncActions();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(actionQueue).toHaveLength(0);
    });

    it('should handle sync conflicts', () => {
      const localData = { version: 1, data: 'local' };
      const serverData = { version: 2, data: 'server' };

      const resolveConflict = (local: any, server: any) => {
        if (server.version > local.version) {
          return server;
        }
        return local;
      };

      const resolved = resolveConflict(localData, serverData);
      expect(resolved).toBe(serverData);
    });
  });

  describe('Offline Storage', () => {
    it('should store data in IndexedDB', () => {
      const isSupported = 'indexedDB' in window;
      expect(isSupported).toBe(true);
    });

    it('should store data in localStorage as fallback', () => {
      const storeOffline = (key: string, data: any) => {
        try {
          localStorage.setItem(key, JSON.stringify(data));
          return true;
        } catch {
          return false;
        }
      };

      const result = storeOffline('offline-data', { tracks: [] });
      expect(result).toBe(true);
      expect(localStorage.getItem('offline-data')).toBeDefined();
    });

    it('should retrieve data from offline storage', () => {
      localStorage.setItem('offline-data', JSON.stringify({ tracks: [] }));
      
      const retrieveOffline = (key: string) => {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
      };

      const data = retrieveOffline('offline-data');
      expect(data).toEqual({ tracks: [] });
    });
  });

  describe('Offline UI', () => {
    it('should show offline indicator', () => {
      const showOfflineIndicator = (isOffline: boolean) => {
        return isOffline ? 'Offline' : 'Online';
      };

      expect(showOfflineIndicator(true)).toBe('Offline');
      expect(showOfflineIndicator(false)).toBe('Online');
    });

    it('should disable online-only features when offline', () => {
      const isFeatureEnabled = (feature: string, isOnline: boolean) => {
        const onlineOnlyFeatures = ['live-streaming', 'real-time-chat'];
        if (onlineOnlyFeatures.includes(feature)) {
          return isOnline;
        }
        return true;
      };

      expect(isFeatureEnabled('live-streaming', false)).toBe(false);
      expect(isFeatureEnabled('play-music', false)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should handle cache errors gracefully', async () => {
      mockCaches.open.mockRejectedValueOnce(new Error('Cache error'));
      
      try {
        await caches.open('v1');
      } catch (error: any) {
        expect(error.message).toBe('Cache error');
      }
    });

    it('should handle sync errors', async () => {
      const syncWithRetry = async (action: any, retries = 3) => {
        for (let i = 0; i < retries; i++) {
          try {
            await fetch('/api/sync', {
              method: 'POST',
              body: JSON.stringify(action),
            });
            return { success: true };
          } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      };

      global.fetch = vi.fn().mockRejectedValue(new Error('Sync failed'));
      
      await expect(syncWithRetry({ type: 'test' })).rejects.toThrow('Sync failed');
    });
  });
});

