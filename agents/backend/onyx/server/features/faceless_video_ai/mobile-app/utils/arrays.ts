/**
 * Array manipulation utilities
 */

export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

export function unique<T>(array: T[]): T[] {
  return Array.from(new Set(array));
}

export function uniqueBy<T>(array: T[], key: keyof T | ((item: T) => unknown)): T[] {
  const seen = new Set<unknown>();
  return array.filter((item) => {
    const value = typeof key === 'function' ? key(item) : item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
}

export function groupBy<T>(
  array: T[],
  key: keyof T | ((item: T) => string)
): Record<string, T[]> {
  return array.reduce((acc, item) => {
    const groupKey =
      typeof key === 'function' ? key(item) : String(item[key]);
    if (!acc[groupKey]) {
      acc[groupKey] = [];
    }
    acc[groupKey].push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

export function sortBy<T>(
  array: T[],
  key: keyof T | ((item: T) => number | string),
  direction: 'asc' | 'desc' = 'asc'
): T[] {
  const sorted = [...array].sort((a, b) => {
    const aValue = typeof key === 'function' ? key(a) : a[key];
    const bValue = typeof key === 'function' ? key(b) : b[key];

    if (aValue < bValue) return direction === 'asc' ? -1 : 1;
    if (aValue > bValue) return direction === 'asc' ? 1 : -1;
    return 0;
  });

  return sorted;
}

export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

export function sample<T>(array: T[], count = 1): T[] {
  const shuffled = shuffle(array);
  return shuffled.slice(0, Math.min(count, array.length));
}

export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce((acc, item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : item);
  }, [] as T[]);
}

export function difference<T>(array1: T[], array2: T[]): T[] {
  return array1.filter((item) => !array2.includes(item));
}

export function intersection<T>(array1: T[], array2: T[]): T[] {
  return array1.filter((item) => array2.includes(item));
}

export function union<T>(array1: T[], array2: T[]): T[] {
  return unique([...array1, ...array2]);
}

export function partition<T>(
  array: T[],
  predicate: (item: T) => boolean
): [T[], T[]] {
  const truthy: T[] = [];
  const falsy: T[] = [];

  array.forEach((item) => {
    if (predicate(item)) {
      truthy.push(item);
    } else {
      falsy.push(item);
    }
  });

  return [truthy, falsy];
}

export function zip<T, U>(array1: T[], array2: U[]): Array<[T, U]> {
  const length = Math.min(array1.length, array2.length);
  const result: Array<[T, U]> = [];

  for (let i = 0; i < length; i++) {
    result.push([array1[i], array2[i]]);
  }

  return result;
}

export function unzip<T, U>(zipped: Array<[T, U]>): [T[], U[]] {
  const array1: T[] = [];
  const array2: U[] = [];

  zipped.forEach(([item1, item2]) => {
    array1.push(item1);
    array2.push(item2);
  });

  return [array1, array2];
}

export function range(start: number, end: number, step = 1): number[] {
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
}

export function compact<T>(array: (T | null | undefined | false | 0 | '')[]): T[] {
  return array.filter((item) => Boolean(item)) as T[];
}

export function take<T>(array: T[], n: number): T[] {
  return array.slice(0, n);
}

export function takeRight<T>(array: T[], n: number): T[] {
  return array.slice(-n);
}

export function drop<T>(array: T[], n: number): T[] {
  return array.slice(n);
}

export function dropRight<T>(array: T[], n: number): T[] {
  return array.slice(0, -n);
}

export function findIndex<T>(
  array: T[],
  predicate: (item: T, index: number) => boolean
): number {
  for (let i = 0; i < array.length; i++) {
    if (predicate(array[i], i)) {
      return i;
    }
  }
  return -1;
}

export function findLastIndex<T>(
  array: T[],
  predicate: (item: T, index: number) => boolean
): number {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i], i)) {
      return i;
    }
  }
  return -1;
}


