'use client';

import { useState, useCallback } from 'react';

export function useArray<T>(initialArray: T[] = []) {
  const [array, setArray] = useState<T[]>(initialArray);

  const push = useCallback((item: T) => {
    setArray((prev) => [...prev, item]);
  }, []);

  const pop = useCallback(() => {
    setArray((prev) => {
      const newArray = [...prev];
      newArray.pop();
      return newArray;
    });
  }, []);

  const shift = useCallback(() => {
    setArray((prev) => {
      const newArray = [...prev];
      newArray.shift();
      return newArray;
    });
  }, []);

  const unshift = useCallback((item: T) => {
    setArray((prev) => [item, ...prev]);
  }, []);

  const remove = useCallback((index: number) => {
    setArray((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const removeByValue = useCallback((value: T) => {
    setArray((prev) => prev.filter((item) => item !== value));
  }, []);

  const update = useCallback((index: number, item: T) => {
    setArray((prev) => {
      const newArray = [...prev];
      newArray[index] = item;
      return newArray;
    });
  }, []);

  const clear = useCallback(() => {
    setArray([]);
  }, []);

  const set = useCallback((newArray: T[]) => {
    setArray(newArray);
  }, []);

  return {
    array,
    set,
    push,
    pop,
    shift,
    unshift,
    remove,
    removeByValue,
    update,
    clear,
    length: array.length,
  };
}

