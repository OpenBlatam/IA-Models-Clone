import { useState, useCallback } from 'react';

export const useList = <T,>(initialList: T[] = []) => {
  const [list, setList] = useState<T[]>(initialList);

  const add = useCallback((item: T): void => {
    setList((prev) => [...prev, item]);
  }, []);

  const remove = useCallback((index: number): void => {
    setList((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const update = useCallback((index: number, item: T): void => {
    setList((prev) => prev.map((el, i) => (i === index ? item : el)));
  }, []);

  const clear = useCallback((): void => {
    setList([]);
  }, []);

  const reset = useCallback((): void => {
    setList(initialList);
  }, [initialList]);

  return {
    list,
    add,
    remove,
    update,
    clear,
    reset,
    setList,
  };
};

