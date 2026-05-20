import { useState, useCallback } from 'react';

export const useMap = <K, V>(initialMap: Map<K, V> = new Map()) => {
  const [map, setMap] = useState<Map<K, V>>(initialMap);

  const set = useCallback((key: K, value: V) => {
    setMap((prev) => {
      const newMap = new Map(prev);
      newMap.set(key, value);
      return newMap;
    });
  }, []);

  const remove = useCallback((key: K) => {
    setMap((prev) => {
      const newMap = new Map(prev);
      newMap.delete(key);
      return newMap;
    });
  }, []);

  const clear = useCallback(() => {
    setMap(new Map());
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

  return {
    map,
    set,
    remove,
    clear,
    get,
    has,
    size: map.size,
    keys: Array.from(map.keys()),
    values: Array.from(map.values()),
    entries: Array.from(map.entries()),
  };
};



