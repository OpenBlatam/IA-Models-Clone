/**
 * Custom hook for cache management.
 * Provides reactive cache with TTL support.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Cache entry with expiration.
 */
interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

/**
 * Options for useCache hook.
 */
export interface UseCacheOptions {
  defaultTTL?: number;
  cleanupInterval?: number;
}

/**
 * Custom hook for cache management.
 * Provides reactive cache with TTL support.
 *
 * @param options - Cache options
 * @returns Cache operations
 */
export function useCache<K, V>(options: UseCacheOptions = {}) {
  const { defaultTTL = 60000, cleanupInterval = 30000 } = options;
  const cacheRef = useRef(new Map<K, CacheEntry<V>>());
  const [, forceUpdate] = useState(0);

  const cleanup = useCallback(() => {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of cacheRef.current.entries()) {
      if (now > entry.expiresAt) {
        cacheRef.current.delete(key);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      forceUpdate((prev) => prev + 1);
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(cleanup, cleanupInterval);
    return () => clearInterval(interval);
  }, [cleanup, cleanupInterval]);

  const set = useCallback(
    (key: K, value: V, ttl?: number) => {
      const expiresAt = Date.now() + (ttl ?? defaultTTL);
      cacheRef.current.set(key, { value, expiresAt });
      forceUpdate((prev) => prev + 1);
    },
    [defaultTTL]
  );

  const get = useCallback((key: K): V | undefined => {
    const entry = cacheRef.current.get(key);
    if (!entry) {
      return undefined;
    }

    if (Date.now() > entry.expiresAt) {
      cacheRef.current.delete(key);
      return undefined;
    }

    return entry.value;
  }, []);

  const has = useCallback(
    (key: K): boolean => {
      return get(key) !== undefined;
    },
    [get]
  );

  const remove = useCallback((key: K): boolean => {
    const deleted = cacheRef.current.delete(key);
    if (deleted) {
      forceUpdate((prev) => prev + 1);
    }
    return deleted;
  }, []);

  const clear = useCallback(() => {
    cacheRef.current.clear();
    forceUpdate((prev) => prev + 1);
  }, []);

  const size = useCallback((): number => {
    return cacheRef.current.size;
  }, []);

  return {
    set,
    get,
    has,
    remove,
    clear,
    size,
    cleanup,
  };
}

