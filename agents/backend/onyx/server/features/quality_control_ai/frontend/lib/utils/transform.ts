export const map = <T, U>(
  array: T[],
  fn: (item: T, index: number) => U
): U[] => {
  return array.map(fn);
};

export const filter = <T,>(
  array: T[],
  fn: (item: T, index: number) => boolean
): T[] => {
  return array.filter(fn);
};

export const reduce = <T, U>(
  array: T[],
  fn: (acc: U, item: T, index: number) => U,
  initialValue: U
): U => {
  return array.reduce(fn, initialValue);
};

export const find = <T,>(
  array: T[],
  fn: (item: T, index: number) => boolean
): T | undefined => {
  return array.find(fn);
};

export const findIndex = <T,>(
  array: T[],
  fn: (item: T, index: number) => boolean
): number => {
  return array.findIndex(fn);
};

export const some = <T,>(
  array: T[],
  fn: (item: T, index: number) => boolean
): boolean => {
  return array.some(fn);
};

export const every = <T,>(
  array: T[],
  fn: (item: T, index: number) => boolean
): boolean => {
  return array.every(fn);
};

export const flatMap = <T, U>(
  array: T[],
  fn: (item: T, index: number) => U | U[]
): U[] => {
  return array.flatMap(fn);
};

export const compact = <T,>(array: (T | null | undefined)[]): T[] => {
  return array.filter((item): item is T => item != null);
};

export const uniq = <T,>(array: T[]): T[] => {
  return Array.from(new Set(array));
};

export const uniqBy = <T, K>(array: T[], keyFn: (item: T) => K): T[] => {
  const seen = new Set<K>();
  return array.filter((item) => {
    const key = keyFn(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
};

export const sortBy = <T,>(
  array: T[],
  keyFn: (item: T) => number | string,
  direction: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aVal = keyFn(a);
    const bVal = keyFn(b);
    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return direction === 'asc' ? comparison : -comparison;
  });
};

export const take = <T,>(array: T[], n: number): T[] => {
  return array.slice(0, n);
};

export const takeRight = <T,>(array: T[], n: number): T[] => {
  return array.slice(-n);
};

export const drop = <T,>(array: T[], n: number): T[] => {
  return array.slice(n);
};

export const dropRight = <T,>(array: T[], n: number): T[] => {
  return array.slice(0, -n);
};

export const chunk = <T,>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

export const zip = <T, U>(array1: T[], array2: U[]): [T, U][] => {
  const length = Math.min(array1.length, array2.length);
  return Array.from({ length }, (_, i) => [array1[i], array2[i]]);
};

export const unzip = <T, U>(array: [T, U][]): [T[], U[]] => {
  return array.reduce(
    (acc, [a, b]) => {
      acc[0].push(a);
      acc[1].push(b);
      return acc;
    },
    [[], []] as [T[], U[]]
  );
};

