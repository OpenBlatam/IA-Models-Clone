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

export const sortByMultiple = <T,>(
  array: T[],
  ...sortFns: Array<(item: T) => number | string>
): T[] => {
  return [...array].sort((a, b) => {
    for (const keyFn of sortFns) {
      const aVal = keyFn(a);
      const bVal = keyFn(b);
      const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      if (comparison !== 0) return comparison;
    }
    return 0;
  });
};

export const sortByNumeric = <T,>(
  array: T[],
  keyFn: (item: T) => number,
  direction: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aVal = keyFn(a);
    const bVal = keyFn(b);
    return direction === 'asc' ? aVal - bVal : bVal - aVal;
  });
};

export const sortByDate = <T,>(
  array: T[],
  keyFn: (item: T) => Date | string | number,
  direction: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aDate = new Date(keyFn(a));
    const bDate = new Date(keyFn(b));
    const comparison = aDate.getTime() - bDate.getTime();
    return direction === 'asc' ? comparison : -comparison;
  });
};

export const sortByLength = <T extends { length: number }>(
  array: T[],
  direction: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const comparison = a.length - b.length;
    return direction === 'asc' ? comparison : -comparison;
  });
};

export const sortAlphabetically = <T,>(
  array: T[],
  keyFn: (item: T) => string,
  direction: 'asc' | 'desc' = 'asc',
  caseSensitive = false
): T[] => {
  return [...array].sort((a, b) => {
    let aVal = keyFn(a);
    let bVal = keyFn(b);

    if (!caseSensitive) {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }

    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return direction === 'asc' ? comparison : -comparison;
  });
};

export const shuffle = <T,>(array: T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

export const reverse = <T,>(array: T[]): T[] => {
  return [...array].reverse();
};

