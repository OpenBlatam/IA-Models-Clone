import { useState, useCallback } from 'react';

export function useMap<K, V>(initialMap: Map<K, V> = new Map()) {
  const [map, setMap] = useState<Map<K, V>>(new Map(initialMap));

  const set = useCallback((key: K, value: V) => {
    setMap((prev) => {
      const next = new Map(prev);
      next.set(key, value);
      return next;
    });
  }, []);

  const get = useCallback(
    (key: K): V | undefined => {
      return map.get(key);
    },
    [map]
  );

  const has = useCallback(
    (key: K): boolean => {
      return map.has(key);
    },
    [map]
  );

  const remove = useCallback((key: K) => {
    setMap((prev) => {
      const next = new Map(prev);
      next.delete(key);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setMap(new Map());
  }, []);

  const setAll = useCallback((newMap: Map<K, V>) => {
    setMap(new Map(newMap));
  }, []);

  return {
    map,
    set,
    get,
    has,
    remove,
    clear,
    setAll,
    size: map.size,
    keys: Array.from(map.keys()),
    values: Array.from(map.values()),
    entries: Array.from(map.entries()),
  };
}



