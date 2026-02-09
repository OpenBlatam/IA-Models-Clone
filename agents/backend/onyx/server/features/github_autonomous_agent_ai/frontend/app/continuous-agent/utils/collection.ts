import type { Array as ArrayUtils } from "./array";
import type { Object as ObjectUtils } from "./object";

export const first = <T>(array: readonly T[]): T | undefined => {
  return array[0];
};

export const last = <T>(array: readonly T[]): T | undefined => {
  return array[array.length - 1];
};

export const findIndex = <T>(
  array: readonly T[],
  predicate: (item: T, index: number) => boolean
): number => {
  return array.findIndex(predicate);
};

export const findLast = <T>(
  array: readonly T[],
  predicate: (item: T, index: number) => boolean
): T | undefined => {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i], i)) {
      return array[i];
    }
  }
  return undefined;
};

export const findLastIndex = <T>(
  array: readonly T[],
  predicate: (item: T, index: number) => boolean
): number => {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i], i)) {
      return i;
    }
  }
  return -1;
};

export const shuffle = <T>(array: readonly T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

export const range = (start: number, end?: number, step: number = 1): number[] => {
  if (end === undefined) {
    end = start;
    start = 0;
  }
  const result: number[] = [];
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
  } else {
    for (let i = start; i > end; i += step) {
      result.push(i);
    }
  }
  return result;
};

export const repeat = <T>(item: T, count: number): T[] => {
  return Array(count).fill(item);
};

export const countBy = <T>(
  array: readonly T[],
  iteratee: (item: T) => string | number
): Record<string, number> => {
  const result: Record<string, number> = {};
  array.forEach((item) => {
    const key = String(iteratee(item));
    result[key] = (result[key] || 0) + 1;
  });
  return result;
};

export const keyBy = <T, K extends string | number>(
  array: readonly T[],
  iteratee: (item: T) => K
): Record<K, T> => {
  const result = {} as Record<K, T>;
  array.forEach((item) => {
    result[iteratee(item)] = item;
  });
  return result;
};

export const sample = <T>(array: readonly T[]): T | undefined => {
  if (array.length === 0) return undefined;
  return array[Math.floor(Math.random() * array.length)];
};

export const sampleSize = <T>(array: readonly T[], size: number): T[] => {
  const shuffled = shuffle(array);
  return shuffled.slice(0, Math.min(size, array.length));
};

export const compact = <T>(array: readonly (T | null | undefined | false | 0 | "")[]): T[] => {
  return array.filter((item): item is T => Boolean(item));
};

export const uniq = <T>(array: readonly T[]): T[] => {
  return Array.from(new Set(array));
};

export const uniqBy = <T, K>(array: readonly T[], iteratee: (item: T) => K): T[] => {
  const seen = new Set<K>();
  return array.filter((item) => {
    const key = iteratee(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
};

export const flatten = <T>(array: readonly (T | T[])[]): T[] => {
  return array.flat();
};

export const flattenDeep = <T>(array: readonly any[]): T[] => {
  return array.flat(Infinity) as T[];
};

export const fromPairs = <K extends string | number, V>(
  pairs: readonly [K, V][]
): Record<K, V> => {
  const result = {} as Record<K, V>;
  pairs.forEach(([key, value]) => {
    result[key] = value;
  });
  return result;
};

export const toPairs = <K extends string | number, V>(
  obj: Record<K, V>
): [K, V][] => {
  return Object.entries(obj) as [K, V][];
};

export const isEmpty = (value: unknown): boolean => {
  if (value == null) return true;
  if (Array.isArray(value) || typeof value === "string") {
    return value.length === 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }
  return false;
};

export const size = (value: unknown): number => {
  if (value == null) return 0;
  if (Array.isArray(value) || typeof value === "string") {
    return value.length;
  }
  if (typeof value === "object") {
    return Object.keys(value).length;
  }
  return 0;
};





