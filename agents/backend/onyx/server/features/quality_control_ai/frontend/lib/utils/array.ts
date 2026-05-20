export const chunk = <T,>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

export const unique = <T,>(array: T[]): T[] => {
  return Array.from(new Set(array));
};

export const uniqueBy = <T, K>(array: T[], keyFn: (item: T) => K): T[] => {
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

export const groupBy = <T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> => {
  return array.reduce(
    (acc, item) => {
      const key = keyFn(item);
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(item);
      return acc;
    },
    {} as Record<K, T[]>
  );
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

export const flatten = <T,>(array: (T | T[])[]): T[] => {
  return array.reduce((acc, item) => {
    return acc.concat(Array.isArray(item) ? flatten(item) : item);
  }, [] as T[]);
};

export const shuffle = <T,>(array: T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

export const sample = <T,>(array: T[], count = 1): T[] => {
  const shuffled = shuffle(array);
  return shuffled.slice(0, Math.min(count, array.length));
};

export const partition = <T,>(array: T[], predicate: (item: T) => boolean): [T[], T[]] => {
  return array.reduce(
    (acc, item) => {
      acc[predicate(item) ? 0 : 1].push(item);
      return acc;
    },
    [[], []] as [T[], T[]]
  );
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

export const intersection = <T,>(array1: T[], array2: T[]): T[] => {
  const set2 = new Set(array2);
  return array1.filter((item) => set2.has(item));
};

export const difference = <T,>(array1: T[], array2: T[]): T[] => {
  const set2 = new Set(array2);
  return array1.filter((item) => !set2.has(item));
};

export const union = <T,>(array1: T[], array2: T[]): T[] => {
  return unique([...array1, ...array2]);
};

