import { useState, useCallback } from 'react';

export const useSet = <T,>(initialSet: Set<T> = new Set()) => {
  const [set, setSet] = useState<Set<T>>(initialSet);

  const add = useCallback((item: T) => {
    setSet((prev) => {
      const newSet = new Set(prev);
      newSet.add(item);
      return newSet;
    });
  }, []);

  const remove = useCallback((item: T) => {
    setSet((prev) => {
      const newSet = new Set(prev);
      newSet.delete(item);
      return newSet;
    });
  }, []);

  const clear = useCallback(() => {
    setSet(new Set());
  }, []);

  const has = useCallback(
    (item: T): boolean => {
      return set.has(item);
    },
    [set]
  );

  return {
    set,
    add,
    remove,
    clear,
    has,
    size: set.size,
    values: Array.from(set.values()),
  };
};



