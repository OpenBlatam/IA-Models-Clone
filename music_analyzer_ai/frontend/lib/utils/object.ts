/**
 * Object utility functions.
 * Provides helper functions for common object operations.
 */

/**
 * Deeply merges two objects.
 * @param target - Target object
 * @param source - Source object to merge
 * @returns Merged object
 */
export function deepMerge<T extends Record<string, unknown>>(
  target: T,
  source: Partial<T>
): T {
  const output = { ...target };

  for (const key in source) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      const sourceValue = source[key];
      const targetValue = output[key];

      if (
        sourceValue &&
        typeof sourceValue === 'object' &&
        !Array.isArray(sourceValue) &&
        targetValue &&
        typeof targetValue === 'object' &&
        !Array.isArray(targetValue)
      ) {
        output[key] = deepMerge(
          targetValue as Record<string, unknown>,
          sourceValue as Record<string, unknown>
        ) as T[Extract<keyof T, string>];
      } else {
        output[key] = sourceValue as T[Extract<keyof T, string>];
      }
    }
  }

  return output;
}

/**
 * Picks specific keys from an object.
 * @param obj - Object to pick from
 * @param keys - Keys to pick
 * @returns Object with only picked keys
 */
export function pick<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;

  keys.forEach((key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
  });

  return result;
}

/**
 * Omits specific keys from an object.
 * @param obj - Object to omit from
 * @param keys - Keys to omit
 * @returns Object without omitted keys
 */
export function omit<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };

  keys.forEach((key) => {
    delete result[key];
  });

  return result;
}

/**
 * Checks if an object is empty.
 * @param obj - Object to check
 * @returns True if object is empty
 */
export function isEmpty(obj: unknown): boolean {
  if (obj === null || obj === undefined) {
    return true;
  }

  if (typeof obj !== 'object') {
    return false;
  }

  if (Array.isArray(obj)) {
    return obj.length === 0;
  }

  return Object.keys(obj).length === 0;
}

/**
 * Gets a nested value from an object using a path string.
 * @param obj - Object to get value from
 * @param path - Dot-separated path (e.g., 'user.profile.name')
 * @param defaultValue - Default value if path doesn't exist
 * @returns Value at path or default value
 */
export function get<T>(
  obj: unknown,
  path: string,
  defaultValue?: T
): T | undefined {
  if (typeof obj !== 'object' || obj === null) {
    return defaultValue;
  }

  const keys = path.split('.');
  let current: unknown = obj;

  for (const key of keys) {
    if (
      current === null ||
      current === undefined ||
      typeof current !== 'object'
    ) {
      return defaultValue;
    }

    current = (current as Record<string, unknown>)[key];
  }

  return (current as T) ?? defaultValue;
}

/**
 * Sets a nested value in an object using a path string.
 * @param obj - Object to set value in
 * @param path - Dot-separated path
 * @param value - Value to set
 * @returns New object with value set
 */
export function set<T extends Record<string, unknown>>(
  obj: T,
  path: string,
  value: unknown
): T {
  const keys = path.split('.');
  const result = { ...obj };
  let current: Record<string, unknown> = result;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!(key in current) || typeof current[key] !== 'object') {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }

  current[keys[keys.length - 1]] = value;

  return result;
}

/**
 * Creates an object from an array of key-value pairs.
 * @param entries - Array of [key, value] pairs
 * @returns Object created from entries
 */
export function fromEntries<T>(
  entries: Array<[string, T]>
): Record<string, T> {
  const result: Record<string, T> = {};

  entries.forEach(([key, value]) => {
    result[key] = value;
  });

  return result;
}

/**
 * Checks if two objects are deeply equal.
 * @param obj1 - First object
 * @param obj2 - Second object
 * @returns True if objects are deeply equal
 */
export function isEqual(obj1: unknown, obj2: unknown): boolean {
  if (obj1 === obj2) {
    return true;
  }

  if (
    obj1 === null ||
    obj2 === null ||
    typeof obj1 !== 'object' ||
    typeof obj2 !== 'object'
  ) {
    return false;
  }

  if (Array.isArray(obj1) && Array.isArray(obj2)) {
    if (obj1.length !== obj2.length) {
      return false;
    }

    for (let i = 0; i < obj1.length; i++) {
      if (!isEqual(obj1[i], obj2[i])) {
        return false;
      }
    }

    return true;
  }

  const keys1 = Object.keys(obj1);
  const keys2 = Object.keys(obj2);

  if (keys1.length !== keys2.length) {
    return false;
  }

  for (const key of keys1) {
    if (!keys2.includes(key)) {
      return false;
    }

    const val1 = (obj1 as Record<string, unknown>)[key];
    const val2 = (obj2 as Record<string, unknown>)[key];

    if (!isEqual(val1, val2)) {
      return false;
    }
  }

  return true;
}

