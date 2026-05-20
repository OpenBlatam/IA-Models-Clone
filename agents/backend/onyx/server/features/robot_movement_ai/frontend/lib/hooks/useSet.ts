import { useState, useCallback } from 'react';

export function useSet<T>(initialSet: Set<T> = new Set()) {
  const [set, setSet] = useState<Set<T>>(new Set(initialSet));

  const add = useCallback((item: T) => {
    setSet((prev) => {
      const next = new Set(prev);
      next.add(item);
      return next;
    });
  }, []);

  const remove = useCallback((item: T) => {
    setSet((prev) => {
      const next = new Set(prev);
      next.delete(item);
      return next;
    });
  }, []);

  const has = useCallback(
    (item: T): boolean => {
      return set.has(item);
    },
    [set]
  );

  const toggle = useCallback((item: T) => {
    setSet((prev) => {
      const next = new Set(prev);
      if (next.has(item)) {
        next.delete(item);
      } else {
        next.add(item);
      }
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setSet(new Set());
  }, []);

  const setAll = useCallback((newSet: Set<T>) => {
    setSet(new Set(newSet));
  }, []);

  return {
    set,
    add,
    remove,
    has,
    toggle,
    clear,
    setAll,
    size: set.size,
    values: Array.from(set.values()),
  };
}



