export const sortByValue = <T extends Record<string, unknown>>(
  obj: T,
  order: 'asc' | 'desc' = 'desc'
): Array<[string, T[keyof T]]> => {
  return Object.entries(obj).sort(([, a], [, b]) => {
    const aVal = typeof a === 'number' ? a : 0;
    const bVal = typeof b === 'number' ? b : 0;
    return order === 'desc' ? bVal - aVal : aVal - bVal;
  });
};

export const extractModelName = (modelPath: string): string => {
  return modelPath.split('/').pop() || modelPath;
};


export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}...`;
};

export const groupBy = <T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<string, T[]> => {
  return array.reduce((acc, item) => {
    const key = String(keyFn(item));
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  }, {} as Record<string, T[]>);
};

