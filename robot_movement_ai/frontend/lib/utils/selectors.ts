/**
 * Selector utilities for data extraction
 */

/**
 * Select property from object
 */
export function select<T, K extends keyof T>(key: K): (obj: T) => T[K] {
  return (obj: T) => obj[key];
}

/**
 * Select nested property
 */
export function selectPath<T>(path: string): (obj: any) => T | undefined {
  return (obj: any) => {
    const keys = path.split('.');
    let current = obj;
    for (const key of keys) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[key];
    }
    return current as T;
  };
}

/**
 * Select multiple properties
 */
export function selectMany<T, K extends keyof T>(
  keys: K[]
): (obj: T) => Pick<T, K> {
  return (obj: T) => {
    const result = {} as Pick<T, K>;
    for (const key of keys) {
      result[key] = obj[key];
    }
    return result;
  };
}

/**
 * Select and transform
 */
export function selectAndTransform<T, K extends keyof T, R>(
  key: K,
  transform: (value: T[K]) => R
): (obj: T) => R {
  return (obj: T) => transform(obj[key]);
}

/**
 * Select with default value
 */
export function selectWithDefault<T, K extends keyof T>(
  key: K,
  defaultValue: T[K]
): (obj: T) => T[K] {
  return (obj: T) => obj[key] ?? defaultValue;
}



