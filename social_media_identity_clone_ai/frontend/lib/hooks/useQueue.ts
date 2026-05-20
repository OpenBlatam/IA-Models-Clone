import { useState, useCallback } from 'react';

interface UseQueueOptions<T> {
  initialValues?: T[];
  limit?: number;
}

export const useQueue = <T,>(options: UseQueueOptions<T> = {}) => {
  const { initialValues = [], limit } = options;
  const [queue, setQueue] = useState<T[]>(initialValues);

  const enqueue = useCallback(
    (item: T) => {
      setQueue((prev) => {
        const newQueue = [...prev, item];
        if (limit && newQueue.length > limit) {
          return newQueue.slice(-limit);
        }
        return newQueue;
      });
    },
    [limit]
  );

  const dequeue = useCallback(() => {
    let item: T | undefined;
    setQueue((prev) => {
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
};



