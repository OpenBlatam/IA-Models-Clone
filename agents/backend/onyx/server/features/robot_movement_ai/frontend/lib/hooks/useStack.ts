import { useState, useCallback } from 'react';

export function useStack<T>(initialStack: T[] = []) {
  const [stack, setStack] = useState<T[]>(initialStack);

  const push = useCallback((item: T) => {
    setStack((prev) => [...prev, item]);
  }, []);

  const pop = useCallback((): T | undefined => {
    let item: T | undefined;
    setStack((prev) => {
      if (prev.length === 0) return prev;
      item = prev[prev.length - 1];
      return prev.slice(0, -1);
    });
    return item;
  }, []);

  const peek = useCallback((): T | undefined => {
    return stack[stack.length - 1];
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
}



