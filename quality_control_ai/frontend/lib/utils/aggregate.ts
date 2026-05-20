export const sum = <T,>(array: T[], keyFn: (item: T) => number): number => {
  return array.reduce((acc, item) => acc + keyFn(item), 0);
};

export const average = <T,>(array: T[], keyFn: (item: T) => number): number => {
  if (array.length === 0) return 0;
  return sum(array, keyFn) / array.length;
};

export const min = <T,>(array: T[], keyFn: (item: T) => number): number | null => {
  if (array.length === 0) return null;
  return Math.min(...array.map(keyFn));
};

export const max = <T,>(array: T[], keyFn: (item: T) => number): number | null => {
  if (array.length === 0) return null;
  return Math.max(...array.map(keyFn));
};

export const count = <T,>(array: T[], predicate?: (item: T) => boolean): number => {
  if (!predicate) return array.length;
  return array.filter(predicate).length;
};

export const countBy = <T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, number> => {
  return array.reduce(
    (acc, item) => {
      const key = keyFn(item);
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    },
    {} as Record<K, number>
  );
};

export const aggregate = <T, R>(
  array: T[],
  initialValue: R,
  reducer: (acc: R, item: T) => R
): R => {
  return array.reduce(reducer, initialValue);
};

export const aggregateBy = <T, K extends string | number, R>(
  array: T[],
  keyFn: (item: T) => K,
  initialValue: R,
  reducer: (acc: R, item: T) => R
): Record<K, R> => {
  return array.reduce(
    (acc, item) => {
      const key = keyFn(item);
      if (!acc[key]) {
        acc[key] = initialValue;
      }
      acc[key] = reducer(acc[key], item);
      return acc;
    },
    {} as Record<K, R>
  );
};

export const median = <T,>(array: T[], keyFn: (item: T) => number): number | null => {
  if (array.length === 0) return null;
  const sorted = [...array].sort((a, b) => keyFn(a) - keyFn(b));
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (keyFn(sorted[mid - 1]) + keyFn(sorted[mid])) / 2;
  }
  return keyFn(sorted[mid]);
};

export const mode = <T,>(array: T[]): T | null => {
  if (array.length === 0) return null;
  const counts = countBy(array, (item) => item as unknown as string | number);
  const maxCount = Math.max(...Object.values(counts));
  const modes = Object.entries(counts)
    .filter(([, count]) => count === maxCount)
    .map(([item]) => item);
  return (modes[0] as unknown as T) || null;
};

