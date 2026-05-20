export const omit = <T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> => {
  const result = { ...obj };
  keys.forEach((key) => {
    delete result[key];
  });
  return result;
};

export const pick = <T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> => {
  const result = {} as Pick<T, K>;
  keys.forEach((key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
  });
  return result;
};

export const keys = <T extends Record<string, unknown>>(obj: T): (keyof T)[] => {
  return Object.keys(obj) as (keyof T)[];
};

export const values = <T extends Record<string, unknown>>(obj: T): T[keyof T][] => {
  return Object.values(obj);
};

export const entries = <T extends Record<string, unknown>>(
  obj: T
): [keyof T, T[keyof T]][] => {
  return Object.entries(obj) as [keyof T, T[keyof T]][];
};

export const isEmpty = (obj: Record<string, unknown>): boolean => {
  return Object.keys(obj).length === 0;
};

export const deepMerge = <T extends Record<string, unknown>>(
  target: T,
  source: Partial<T>
): T => {
  const output = { ...target };
  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach((key) => {
      const sourceValue = source[key];
      const targetValue = target[key];
      if (isObject(sourceValue) && isObject(targetValue)) {
        output[key as keyof T] = deepMerge(
          targetValue as Record<string, unknown>,
          sourceValue as Record<string, unknown>
        ) as T[keyof T];
      } else {
        output[key as keyof T] = sourceValue as T[keyof T];
      }
    });
  }
  return output;
};

const isObject = (item: unknown): item is Record<string, unknown> => {
  return item !== null && typeof item === 'object' && !Array.isArray(item);
};

export const mapValues = <T, U>(
  obj: Record<string, T>,
  fn: (value: T, key: string) => U
): Record<string, U> => {
  const result: Record<string, U> = {};
  Object.keys(obj).forEach((key) => {
    result[key] = fn(obj[key], key);
  });
  return result;
};

export const invert = <T extends string | number>(
  obj: Record<string, T>
): Record<T, string> => {
  const result: Record<T, string> = {} as Record<T, string>;
  Object.keys(obj).forEach((key) => {
    result[obj[key]] = key;
  });
  return result;
};

