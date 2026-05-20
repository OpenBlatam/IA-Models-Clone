/**
 * Data transformation utilities
 * 
 * Provides functions to transform data structures
 */

/**
 * Transforms an array of objects into a map by key
 */
export function arrayToMap<T, K extends keyof T>(
  array: T[],
  key: K
): Map<T[K] & (string | number), T> {
  const map = new Map<T[K] & (string | number), T>();
  for (const item of array) {
    map.set(item[key] as T[K] & (string | number), item);
  }
  return map;
}

/**
 * Transforms an array of objects into a record by key
 */
export function arrayToRecord<T, K extends keyof T>(
  array: T[],
  key: K
): Record<string, T> {
  const record: Record<string, T> = {};
  for (const item of array) {
    const keyValue = String(item[key]);
    record[keyValue] = item;
  }
  return record;
}

/**
 * Groups array items by a key
 */
export function groupBy<T, K extends keyof T>(
  array: T[],
  key: K
): Map<T[K] & (string | number), T[]> {
  const map = new Map<T[K] & (string | number), T[]>();
  for (const item of array) {
    const keyValue = item[key] as T[K] & (string | number);
    const group = map.get(keyValue) ?? [];
    group.push(item);
    map.set(keyValue, group);
  }
  return map;
}

/**
 * Flattens a nested array
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce<T[]>((acc, item) => {
    if (Array.isArray(item)) {
      acc.push(...item);
    } else {
      acc.push(item);
    }
    return acc;
  }, []);
}

/**
 * Flattens a deeply nested array
 */
export function flattenDeep<T>(array: unknown[]): T[] {
  return array.reduce<T[]>((acc, item) => {
    if (Array.isArray(item)) {
      acc.push(...flattenDeep<T>(item));
    } else {
      acc.push(item as T);
    }
    return acc;
  }, []);
}

/**
 * Picks specified keys from an object
 */
export function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  return result;
}

/**
 * Omits specified keys from an object
 */
export function omit<T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
  const result = { ...obj };
  for (const key of keys) {
    delete result[key];
  }
  return result;
}

/**
 * Maps object values
 */
export function mapValues<T, R>(
  obj: Record<string, T>,
  fn: (value: T, key: string) => R
): Record<string, R> {
  const result: Record<string, R> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[key] = fn(value, key);
  }
  return result;
}

/**
 * Maps object keys
 */
export function mapKeys<T>(
  obj: Record<string, T>,
  fn: (key: string, value: T) => string
): Record<string, T> {
  const result: Record<string, T> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[fn(key, value)] = value;
  }
  return result;
}

/**
 * Inverts object keys and values
 */
export function invert<T extends string | number>(
  obj: Record<string, T>
): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[String(value)] = key;
  }
  return result;
}




