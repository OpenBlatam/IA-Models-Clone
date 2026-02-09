'use client';

import { useState, useCallback } from 'react';
import { cacheManager } from '@/lib/cache-manager';

export function useCache<T>(key: string, ttl?: number) {
  const [data, setData] = useState<T | null>(() => cacheManager.get<T>(key));

  const set = useCallback(
    (value: T) => {
      cacheManager.set(key, value, ttl);
      setData(value);
    },
    [key, ttl]
  );

  const get = useCallback(() => {
    const cached = cacheManager.get<T>(key);
    setData(cached);
    return cached;
  }, [key]);

  const remove = useCallback(() => {
    cacheManager.delete(key);
    setData(null);
  }, [key]);

  const has = useCallback(() => {
    return cacheManager.has(key);
  }, [key]);

  return {
    data,
    set,
    get,
    remove,
    has,
  };
}

