/**
 * State management helpers
 */

/**
 * Create state updater that merges with previous state
 */
export function createMergedUpdater<T extends Record<string, any>>() {
  return (prev: T, updates: Partial<T>): T => ({
    ...prev,
    ...updates,
  });
}

/**
 * Create state updater for arrays (add item)
 */
export function createArrayAdder<T>() {
  return (prev: T[], item: T): T[] => [...prev, item];
}

/**
 * Create state updater for arrays (remove item)
 */
export function createArrayRemover<T>() {
  return (prev: T[], item: T): T[] => prev.filter((i) => i !== item);
}

/**
 * Create state updater for arrays (update item)
 */
export function createArrayUpdater<T>(
  findFn: (item: T) => boolean
) {
  return (prev: T[], updater: (item: T) => T): T[] =>
    prev.map((item) => (findFn(item) ? updater(item) : item));
}

/**
 * Create state updater for maps
 */
export function createMapUpdater<K, V>() {
  return (prev: Map<K, V>, key: K, value: V): Map<K, V> => {
    const next = new Map(prev);
    next.set(key, value);
    return next;
  };
}

/**
 * Create state updater for sets
 */
export function createSetUpdater<T>() {
  return {
    add: (prev: Set<T>, item: T): Set<T> => {
      const next = new Set(prev);
      next.add(item);
      return next;
    },
    remove: (prev: Set<T>, item: T): Set<T> => {
      const next = new Set(prev);
      next.delete(item);
      return next;
    },
    toggle: (prev: Set<T>, item: T): Set<T> => {
      const next = new Set(prev);
      if (next.has(item)) {
        next.delete(item);
      } else {
        next.add(item);
      }
      return next;
    },
  };
}



