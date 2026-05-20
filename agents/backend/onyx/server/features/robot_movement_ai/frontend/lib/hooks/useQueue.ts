import { useState, useCallback } from 'react';

export function useQueue<T>(initialQueue: T[] = []) {
  const [queue, setQueue] = useState<T[]>(initialQueue);

  const enqueue = useCallback((item: T) => {
    setQueue((prev) => [...prev, item]);
  }, []);

  const dequeue = useCallback((): T | undefined => {
    let item: T | undefined;
    setQueue((prev) => {
      if (prev.length === 0) return prev;
      item = prev[0];
      return prev.slice(1);
    });
    return item;
  }, []);

  const peek = useCallback((): T | undefined => {
    return queue[0];
  }, [queue]);

  const clear = useCallback(() => {
    setQueue([]);
  }, []);

  return {
    queue,
    enqueue,
    dequeue,
    peek,
    clear,
    size: queue.length,
    isEmpty: queue.length === 0,
  };
}



