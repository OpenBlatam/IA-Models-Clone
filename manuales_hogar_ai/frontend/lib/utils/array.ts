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

export const sortBy = <T,>(
  array: T[],
  keyFn: (item: T) => number | string,
  order: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aVal = keyFn(a);
    const bVal = keyFn(b);
    if (aVal < bVal) return order === 'asc' ? -1 : 1;
    if (aVal > bVal) return order === 'asc' ? 1 : -1;
    return 0;
  });
};

export const take = <T,>(array: T[], n: number): T[] => {
  return array.slice(0, n);
};

export const takeWhile = <T,>(array: T[], predicate: (item: T) => boolean): T[] => {
  const result: T[] = [];
  for (const item of array) {
    if (!predicate(item)) break;
    result.push(item);
  }
  return result;
};

