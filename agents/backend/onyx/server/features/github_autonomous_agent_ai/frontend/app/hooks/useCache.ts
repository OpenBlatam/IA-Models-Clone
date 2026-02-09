/**
 * Hook para cache en memoria con TTL y persistencia.
 */

import { useState, useCallback, useEffect } from 'react';

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  createdAt: number;
}

interface UseCacheOptions {
  ttl?: number; // TTL en milisegundos
  persist?: boolean; // Persistir en localStorage
  storageKey?: string; // Key para localStorage
}

/**
 * Hook para cache con TTL.
 */
export function useCache<T>(options: UseCacheOptions = {}) {
  const {
    ttl = 5 * 60 * 1000, // 5 minutos por defecto
    persist = false,
    storageKey = 'app_cache'
  } = options;

  const [cache, setCache] = useState<Map<string, CacheEntry<T>>>(new Map());

  // Cargar desde localStorage al montar
  useEffect(() => {
    if (persist && typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(storageKey);
        if (stored) {
          const parsed = JSON.parse(stored);
          const now = Date.now();
          const validEntries = new Map<string, CacheEntry<T>>();
          
          for (const [key, entry] of Object.entries(parsed)) {
            const cacheEntry = entry as CacheEntry<T>;
            if (cacheEntry.expiresAt > now) {
              validEntries.set(key, cacheEntry);
            }
          }
          
          setCache(validEntries);
        }
      } catch (error) {
        console.error('Error loading cache:', error);
      }
    }
  }, [persist, storageKey]);

  // Guardar en localStorage cuando cambia
  useEffect(() => {
    if (persist && typeof window !== 'undefined') {
      try {
        const toStore: Record<string, CacheEntry<T>> = {};
        cache.forEach((value, key) => {
          toStore[key] = value;
        });
        localStorage.setItem(storageKey, JSON.stringify(toStore));
      } catch (error) {
        console.error('Error saving cache:', error);
      }
    }
  }, [cache, persist, storageKey]);

  const get = useCallback((key: string): T | null => {
    const entry = cache.get(key);
    if (!entry) return null;

    const now = Date.now();
    if (entry.expiresAt <= now) {
      // Expiró, remover
      setCache(prev => {
        const newCache = new Map(prev);
        newCache.delete(key);
        return newCache;
      });
      return null;
    }

    return entry.value;
  }, [cache]);

  const set = useCallback((key: string, value: T, customTtl?: number) => {
    const now = Date.now();
    const expiresAt = now + (customTtl || ttl);

    setCache(prev => {
      const newCache = new Map(prev);
      newCache.set(key, {
        value,
        expiresAt,
        createdAt: now
      });
      return newCache;
    });
  }, [ttl]);

  const remove = useCallback((key: string) => {
    setCache(prev => {
      const newCache = new Map(prev);
      newCache.delete(key);
      return newCache;
    });
  }, []);

  const clear = useCallback(() => {
    setCache(new Map());
    if (persist && typeof window !== 'undefined') {
      localStorage.removeItem(storageKey);
    }
  }, [persist, storageKey]);

  const has = useCallback((key: string): boolean => {
    const entry = cache.get(key);
    if (!entry) return false;
    return entry.expiresAt > Date.now();
  }, [cache]);

  // Limpiar entradas expiradas periódicamente
  useEffect(() => {
    const interval = setInterval(() => {
      setCache(prev => {
        const now = Date.now();
        const newCache = new Map(prev);
        for (const [key, entry] of newCache.entries()) {
          if (entry.expiresAt <= now) {
            newCache.delete(key);
          }
        }
        return newCache;
      });
    }, 60000); // Cada minuto

    return () => clearInterval(interval);
  }, []);

  return {
    get,
    set,
    remove,
    clear,
    has,
    size: cache.size
  };
}



