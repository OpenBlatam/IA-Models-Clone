import { useState, useCallback } from 'react';

export const useSet = <T,>(initialSet: Set<T> | T[] = []) => {
  const [set, setSet] = useState<Set<T>>(() => {
    if (initialSet instanceof Set) {
      return new Set(initialSet);
    }
    return new Set(initialSet);
  });

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

  const clear = useCallback(() => {
    setSet(new Set());
  }, []);

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

  return {
    set,
    add,
    remove,
    has,
    clear,
    toggle,
    size: set.size,
    isEmpty: set.size === 0,
    values: Array.from(set.values()),
  };
};

