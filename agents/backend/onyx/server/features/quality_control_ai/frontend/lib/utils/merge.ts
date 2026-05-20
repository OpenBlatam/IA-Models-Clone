export const merge = <T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T => {
  return sources.reduce((acc, source) => {
    return { ...acc, ...source };
  }, target);
};

export const deepMerge = <T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T => {
  return sources.reduce((acc, source) => {
    const result = { ...acc };

    Object.keys(source).forEach((key) => {
      const sourceValue = source[key];
      const targetValue = result[key];

      if (
        typeof sourceValue === 'object' &&
        sourceValue !== null &&
        !Array.isArray(sourceValue) &&
        typeof targetValue === 'object' &&
        targetValue !== null &&
        !Array.isArray(targetValue)
      ) {
        result[key] = deepMerge(
          targetValue as Record<string, unknown>,
          sourceValue as Record<string, unknown>
        ) as T[Extract<keyof T, string>];
      } else {
        result[key] = sourceValue as T[Extract<keyof T, string>];
      }
    });

    return result;
  }, target);
};

export const mergeArrays = <T,>(...arrays: T[][]): T[] => {
  return arrays.reduce((acc, arr) => [...acc, ...arr], []);
};

export const mergeUnique = <T,>(...arrays: T[][]): T[] => {
  return Array.from(new Set(mergeArrays(...arrays)));
};

export const mergeBy = <T, K>(
  arrays: T[][],
  keyFn: (item: T) => K
): T[] => {
  const map = new Map<K, T>();

  arrays.forEach((array) => {
    array.forEach((item) => {
      const key = keyFn(item);
      if (!map.has(key)) {
        map.set(key, item);
      }
    });
  });

  return Array.from(map.values());
};

