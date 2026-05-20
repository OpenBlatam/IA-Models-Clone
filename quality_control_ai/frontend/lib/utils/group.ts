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

export const groupByMultiple = <T, K extends string | number>(
  array: T[],
  ...keyFns: Array<(item: T) => K>
): Record<string, T[]> => {
  return array.reduce(
    (acc, item) => {
      const keys = keyFns.map((fn) => String(fn(item))).join('|');
      if (!acc[keys]) {
        acc[keys] = [];
      }
      acc[keys].push(item);
      return acc;
    },
    {} as Record<string, T[]>
  );
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

export const chunk = <T,>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

export const split = <T,>(array: T[], index: number): [T[], T[]] => {
  return [array.slice(0, index), array.slice(index)];
};

export const splitBy = <T,>(array: T[], predicate: (item: T) => boolean): [T[], T[]] => {
  const index = array.findIndex(predicate);
  if (index === -1) return [array, []];
  return split(array, index);
};

export const groupByCount = <T,>(
  array: T[],
  count: number
): T[][] => {
  return chunk(array, count);
};

