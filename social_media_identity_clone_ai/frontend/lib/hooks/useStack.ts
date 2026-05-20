import { useState, useCallback } from 'react';

interface UseStackOptions<T> {
  initialValues?: T[];
  limit?: number;
}

export const useStack = <T,>(options: UseStackOptions<T> = {}) => {
  const { initialValues = [], limit } = options;
  const [stack, setStack] = useState<T[]>(initialValues);

  const push = useCallback(
    (item: T) => {
      setStack((prev) => {
        const newStack = [item, ...prev];
        if (limit && newStack.length > limit) {
          return newStack.slice(0, limit);
        }
        return newStack;
      });
    },
    [limit]
  );

  const pop = useCallback(() => {
    let item: T | undefined;
    setStack((prev) => {
      if (prev.length === 0) {
        return prev;
      }
      const [first, ...rest] = prev;
      item = first;
      return rest;
    });
    return item;
  }, []);

  const peek = useCallback((): T | undefined => {
    return stack[0];
  }, [stack]);

  const clear = useCallback(() => {
    setStack([]);
  }, []);

  return {
    stack,
    push,
    pop,
    peek,
    clear,
    size: stack.length,
    isEmpty: stack.length === 0,
  };
};



